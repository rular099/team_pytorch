"""Stdlib proof of the operations loop's narrowly scoped source delta.

The authenticated engine, gates and checkpoint identity stay immutable. The
separate executed loop and its CPU-RNG compatibility change have their own SHA.
No code generation or global monkeypatch is used by the launcher.
"""
import ast
import copy
import dis
import hashlib
from pathlib import Path
from types import CodeType


RNG_OLD = "torch.cuda.set_rng_state_all(rng['cuda_rng'])"
RNG_NEW = "torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda_rng']])"
FIRST_BATCH = 'micro == 0 and epoch == start_epoch'
STAGES = {
    'validate.begin', 'validate.complete', 'distributed_init.begin',
    'distributed_init.complete', 'initial_barrier.begin', 'initial_barrier.complete',
    'dataset.begin', 'dataset.complete', 'model_build.begin', 'model_build.complete',
    'audited_initial_state.begin', 'audited_initial_state.complete',
    'ddp_constructor.begin', 'ddp_constructor.complete', 'optimizer.begin',
    'optimizer.complete', 'checkpoint_load.begin', 'checkpoint_load.complete',
    'checkpoint_model_restore.begin', 'checkpoint_model_restore.complete',
    'optimizer_restore.begin', 'optimizer_restore.complete',
    'rng_restore.begin', 'rng_restore.complete', 'init_publish.begin',
    'init_publish.complete', 'training_loop.enter', 'first_batch.ready',
    'first_update.complete', 'training.complete',
}


def dump(node):
    return ast.dump(node, include_attributes=False)


def verify(root, compatibility_file=None):
    """Fail closed unless only CPU conversion and allowed observations differ."""
    original_path = Path(root)/'fe01_a01/engine.py'
    path = Path(compatibility_file or Path(__file__).with_name('training_compat.py'))
    original_source, adapted_source = original_path.read_text(), path.read_text()
    original_tree, adapted_tree = ast.parse(original_source), ast.parse(adapted_source)
    original = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    adapted = next(n for n in adapted_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    # The copied loop imports its exact dependencies from the immutable engine.
    allowed = (ast.ImportFrom, ast.FunctionDef)
    for node in adapted_tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, allowed):
            raise ValueError('Compatibility module has unexpected top-level code')
        if isinstance(node, ast.ImportFrom) and (node.module != 'fe01_a01.engine' or node.level or
                                                any(a.asname or a.name == '*' for a in node.names)):
            raise ValueError('Compatibility dependencies must be explicit original engine bindings')
        if isinstance(node, ast.FunctionDef) and node.name != 'train':
            raise ValueError('Unexpected compatibility function')
    proof = copy.deepcopy(adapted)
    if len(proof.args.kwonlyargs) != 1 or proof.args.kwonlyargs[0].arg != 'observe' or proof.args.kw_defaults != [None]:
        raise ValueError('Compatibility loop requires one explicit observer')
    proof.args.kwonlyargs = [];proof.args.kw_defaults = []
    old_expr, new_expr = ast.parse(RNG_OLD).body[0], ast.parse(RNG_NEW).body[0]
    count = [0]

    class Normalize(ast.NodeTransformer):
        def visit_Expr(self, node):
            if dump(node) == dump(new_expr):
                count[0] += 1
                return copy.deepcopy(old_expr)
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'observe':
                if (len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Constant)
                        or call.args[0].value not in STAGES):
                    raise ValueError('Unexpected observer expression')
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            # Only this added conditional may disappear after removing observers.
            before = dump(node.test)
            self.generic_visit(node)
            if not node.body:
                if before != dump(ast.parse(FIRST_BATCH, mode='eval').body) or node.orelse:
                    raise ValueError('Unexpected empty observer guard')
                return None
            return node

    proof = Normalize().visit(proof)
    if count[0] != 1 or dump(proof) != dump(original):
        raise ValueError('Compatibility loop differs beyond CPU RNG conversion and startup observations')
    return dict(
        original_engine_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        original_train_ast_sha256=hashlib.sha256(dump(original).encode()).hexdigest(),
        executed_compatibility_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        allowed_semantic_delta=RNG_NEW, observer_stages=sorted(STAGES),
        source_delta_check='PASS', original_gates_unchanged=True,
    )


def verify_bindings(engine, compatibility):
    """Prevent the copied body from silently calling substituted dependencies."""
    def globals_used(code):
        names = {i.argval for i in dis.get_instructions(code) if i.opname == 'LOAD_GLOBAL'}
        for value in code.co_consts:
            if isinstance(value, CodeType):
                names.update(globals_used(value))
        return names
    for name in globals_used(engine.train.__code__):
        if name in engine.train.__globals__:
            if compatibility.train.__globals__.get(name) is not engine.train.__globals__[name]:
                raise ValueError('Compatibility dependency differs: '+name)
