#!/usr/bin/env bash
# Standalone operations launcher. Does not edit authenticated A01 source/gates.
# LOGIN: inspect [ARRAY_JOB_ID] | probe | retry FAILED_INDICES_OR_failed
set -euo pipefail
self=$(readlink -f "${BASH_SOURCE[0]}")
tool_dir=$(dirname "$self")
mode=${1:-inspect}
private=${A01_ENV_FILE:-}
if [[ -z "$private" ]]; then
  for candidate in "$tool_dir/a01.private.env" "$tool_dir/.a01_releases/3ecf060a3334/a01.private.env"; do
    if [[ -f "$candidate" ]]; then private=$candidate;break;fi
  done
fi
[[ -f "$private" ]] || { echo '找不到私有设置；设置 A01_ENV_FILE 为正在使用的 a01.private.env 绝对路径。' >&2;exit 2; }
export A01_ENV_FILE=$(readlink -f "$private")
source "$A01_ENV_FILE"
: "${A01_CODE_ROOT:?}" "${A01_OUTPUT_ROOT:?}" "${A01_TRAIN_NODES:?}" "${A01_DEVICES_PER_NODE:?}"
[[ -f "$A01_CODE_ROOT/scripts/fe01_a01/preflight.py" ]] || { echo '需要已部署的 backend_fix A01 release。' >&2;exit 2; }
export A01_WORLD_SIZE=$((A01_TRAIN_NODES*A01_DEVICES_PER_NODE))
[[ "$A01_WORLD_SIZE" == 16 ]] || { echo '保持已认证的 world16；请核对私有资源设置。' >&2;exit 2; }

scan() {
  "${A01_CONFIG_PYTHON:-python3}" - "$A01_OUTPUT_ROOT" "${1:-}" "${2:-json}" <<'PY'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]);job=sys.argv[2]
if not job:
    jobs=root/'jobs.tsv'
    rows=[row.split('\t') for row in jobs.read_text().splitlines() if row.startswith('train\t')] if jobs.is_file() else []
    if not rows:raise SystemExit('没有旧 train 作业记录；请用 inspect ARRAY_JOB_ID 指定作业号。')
    job=rows[-1][2].split(';')[0]
if not re.fullmatch(r'[0-9]+',job):raise SystemExit('数组作业号须为纯数字。')
rows=[];failed=[]
for index,family in enumerate(('TEAM OFF','PhaseNet OFF','EQT OFF','DiTing ON','DiTing OFF')):
    path=root/'slurm'/('train_'+job+'_'+str(index)+'.err')
    text=path.read_text(errors='replace') if path.is_file() else ''
    match='AssertionError: DCU unavailable in allocation' in text
    if match:failed.append(str(index))
    rows.append(dict(index=index,family=family,log_present=path.is_file(),startup_DCU_assertion=match))
if sys.argv[3]=='indices':
    if not failed:raise SystemExit('未找到这批 DCU 启动断言；不会猜测失败项或重提成功项。')
    print(','.join(failed))
else:print(json.dumps(dict(array_job_id=job,failed_indices=failed,logs=rows,note='无断言不等于训练完成；inspect只读，不提交。'),ensure_ascii=False,indent=2))
PY
}

# EXIT trap preserves the original env.sh error and additionally records HIP init.
context_exit() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ -n "${A01_PYTHON:-}" && -x "$A01_PYTHON" ]]; then
    "$A01_PYTHON" - "$rc" <<'PY'
import glob,json,os,socket,sys
record=dict(context=os.environ.get('A01_PROBE_CONTEXT'),host=socket.gethostname(),environment_exit=int(sys.argv[1]),
    visibility={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_STEP_ID','SLURM_PROCID','SLURM_LOCALID',
    'SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')},
    kfd_present=os.path.exists('/dev/kfd'),kfd_access=os.access('/dev/kfd',os.R_OK|os.W_OK),
    render_devices=glob.glob('/dev/dri/renderD*'),status='FAIL')
try:
    import torch
    record.update(torch=torch.__version__,rocm=torch.version.hip,available=torch.cuda.is_available(),count=torch.cuda.device_count())
    torch.cuda.init()
    local=int(os.environ.get('SLURM_LOCALID','0')) if record['context']=='step' else 0
    torch.cuda.set_device(local)
    value=torch.ones(1,device='cuda:'+str(local));record['tiny_device_check']=float((value+1).item())
    record['device']=torch.cuda.get_device_name(local)
    record['status']='PASS' if record['environment_exit']==0 and record['available'] and record['tiny_device_check']==2. else 'FAIL'
except Exception as error:record['init_error']=type(error).__name__+': '+str(error)
print('A01_DCU_PROBE '+json.dumps(record,ensure_ascii=False),flush=True)
raise SystemExit(0 if record['status']=='PASS' else 1)
PY
    local probe_rc=$?
    if [[ "$rc" == 0 ]]; then rc=$probe_rc;fi
  else
    printf 'A01_DCU_PROBE_NO_PYTHON host=%s rc=%s\n' "$(hostname)" "$rc"
    [[ "$rc" != 0 ]] || rc=2
  fi
  exit "$rc"
}

case "$mode" in
  _context)
    : "${SLURM_JOB_ID:?Compute allocation required}"
    export A01_PROBE_CONTEXT=${2:?batch|step}
    printf 'A01_DCU_CONTEXT context=%s host=%s rank=%s\n' "$A01_PROBE_CONTEXT" "$(hostname)" "${SLURM_PROCID:-batch}"
    trap context_exit EXIT
    source "$A01_CODE_ROOT/scripts/fe01_a01/env.sh"
    exit 0
    ;;
  _rank_train)
    : "${SLURM_JOB_ID:?}"
    source "$A01_CODE_ROOT/scripts/fe01_a01/env.sh"
    index=${SLURM_ARRAY_TASK_ID:?}
    exec bash "$A01_CODE_ROOT/scripts/fe01_a01/rank_exec.sh" "$A01_PYTHON" \
      scripts/fe01_a01/run.py train --index "$index" --output "$A01_OUTPUT_ROOT" --device cuda \
      --world "$A01_WORLD_SIZE" --probe-cap "${A01_PROBE_CAP:-2}"
    ;;
  _job_probe|_job_retry)
    : "${SLURM_JOB_ID:?}"
    [[ "${SLURM_JOB_NUM_NODES:?}" == "$A01_TRAIN_NODES" ]] || { echo '实际节点数与已认证资源不符。' >&2;exit 2; }
    step=(srun --nodes="$A01_TRAIN_NODES" --ntasks="$A01_WORLD_SIZE" --ntasks-per-node="$A01_DEVICES_PER_NODE" --gres="$A01_GRES")
    if [[ "$mode" == _job_probe ]]; then
      set +e
      bash "$self" _context batch
      batch_rc=$?
      "${step[@]}" --kill-on-bad-exit=0 bash "$self" _context step
      step_rc=$?
      printf 'A01_DCU_SUMMARY batch_rc=%s step_rc=%s (step0 only means this allocation passed)\n' "$batch_rc" "$step_rc"
      exit "$step_rc"
    fi
    # Every actual step rank must pass before creating a training run directory.
    "${step[@]}" --kill-on-bad-exit=1 bash "$self" _context step
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=$((A01_MASTER_PORT+SLURM_ARRAY_TASK_ID))
    exec "${step[@]}" --kill-on-bad-exit=1 bash "$self" _rank_train
    ;;
  inspect)
    scan "${2:-}"
    exit 0
    ;;
  probe) indices='';;
  retry)
    indices=${2:?Usage: retry failed OR retry 0,2,4; only selected empty failed runs}
    [[ "$indices" != failed ]] || indices=$(scan "" indices)
    [[ "$indices" =~ ^[0-4](,[0-4])*$ ]] || { echo '使用逗号分隔的索引，如0,2,4；或failed自动识别旧启动断言。' >&2;exit 2; }
    IFS=, read -r -a chosen <<< "$indices"
    declare -A seen=()
    for index in "${chosen[@]}"; do
      [[ -z "${seen[$index]:-}" ]] || { echo '重复索引。' >&2;exit 2; }
      seen[$index]=1
      "${A01_CONFIG_PYTHON:-python3}" "$A01_CODE_ROOT/scripts/fe01_a01/preflight.py" train --indices "$index" --output "$A01_OUTPUT_ROOT"
    done
    ;;
  *) echo 'Usage: inspect [ARRAY_JOB_ID] | probe | retry failed | retry 0,2,4' >&2;exit 2;;
esac

# This part runs only for a user-invoked probe/retry, never for inspect/compute.
mkdir -p "$A01_OUTPUT_ROOT/operations"
operation=$(mktemp -d "$A01_OUTPUT_ROOT/operations/dcu_${mode}_$(date +%Y%m%dT%H%M%S)_XXXXXX")
cp "$self" "$operation/dcu_recovery.sh"
cp "$A01_ENV_FILE" "$operation/a01.private.env"
chmod 500 "$operation/dcu_recovery.sh"
chmod 600 "$operation/a01.private.env"
export A01_ENV_FILE="$operation/a01.private.env"
sha256sum "$operation/dcu_recovery.sh" > "$operation/launcher.sha256"
sha256sum "$operation/a01.private.env" > "$operation/private_settings.sha256"
if [[ -f "$A01_CODE_ROOT/FE01_A01_SOURCE_IDENTITY.json" ]]; then
  cp "$A01_CODE_ROOT/FE01_A01_SOURCE_IDENTITY.json" "$operation/core_source_identity.json"
fi
printf 'mode=%s\nindices=%s\ncode=%s\noutput=%s\nnodes=%s\ndevices_per_node=%s\nworld=%s\ngres=%s\nexclude=%s\n' \
  "$mode" "$indices" "$A01_CODE_ROOT" "$A01_OUTPUT_ROOT" "$A01_TRAIN_NODES" "$A01_DEVICES_PER_NODE" \
  "$A01_WORLD_SIZE" "$A01_GRES" "${A01_EXCLUDE_NODES:-}" > "$operation/request.txt"
args=(--parsable --chdir="$A01_CODE_ROOT" --export=ALL --job-name="a01_dcu_$mode" \
  --nodes="$A01_TRAIN_NODES" --ntasks-per-node="$A01_DEVICES_PER_NODE" --cpus-per-task="$A01_CPUS_PER_TASK" \
  --gres="$A01_GRES" --mem="${A01_TRAIN_MEM:-64G}" --output="$operation/%A_%a.out" --error="$operation/%A_%a.err")
[[ -z "${A01_PARTITION:-}" ]] || args+=(--partition="$A01_PARTITION")
[[ -z "${A01_EXCLUDE_NODES:-}" ]] || args+=(--exclude="$A01_EXCLUDE_NODES")
if [[ "$mode" == probe ]]; then
  args+=(--time=00:10:00);action=_job_probe
else
  args+=(--time="$A01_TRAIN_TIME" --array="$indices%3");action=_job_retry
fi
job=$(sbatch "${args[@]}" "$operation/dcu_recovery.sh" "$action")
printf '%s\t%s\t%s\n' "$mode" "$indices" "$job" > "$operation/job.tsv"
printf 'SUBMITTED mode=%s indices=%s job=%s\nLogs=%s\n' "$mode" "$indices" "$job" "$operation"
