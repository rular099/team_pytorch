import os
import tempfile
import unittest

import torch

from train_light import load_checkpoint, save_model_checkpoint


def _training_objects(lr=1.0):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=0,
    )
    return model, optimizer, scheduler


class SchedulerCheckpointResumeTests(unittest.TestCase):
    def test_legacy_pre_step_checkpoint_replays_plateau_observation(self):
        model, optimizer, scheduler = _training_objects()
        scheduler.step(1.0)
        _, expected_optimizer, expected_scheduler = _training_objects()
        expected_scheduler.step(1.0)
        expected_scheduler.step(2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'legacy.pth')
            torch.save(
                {
                    'epoch': 2,
                    'loss': 2.0,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'excluded_prefixes': [],
                },
                checkpoint_path,
            )
            restored_model, restored_optimizer, restored_scheduler = _training_objects()
            _, restored_optimizer, restored_scheduler, epoch, loss = load_checkpoint(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                checkpoint_path,
                device='cpu',
                lr_monitor='val',
            )

        self.assertEqual(epoch, 2)
        self.assertEqual(loss, 2.0)
        self.assertAlmostEqual(restored_optimizer.param_groups[0]['lr'], 0.5)
        self.assertAlmostEqual(restored_scheduler.best, 1.0)
        self.assertEqual(restored_scheduler.state_dict(), expected_scheduler.state_dict())
        self.assertEqual(
            restored_optimizer.param_groups[0]['lr'],
            expected_optimizer.param_groups[0]['lr'],
        )

    def test_legacy_pre_step_checkpoint_replays_new_best_value(self):
        model, optimizer, scheduler = _training_objects()
        scheduler.step(1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'legacy-new-best.pth')
            torch.save(
                {
                    'epoch': 20,
                    'loss': 0.5,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'excluded_prefixes': [],
                },
                checkpoint_path,
            )
            restored_model, restored_optimizer, restored_scheduler = _training_objects()
            _, restored_optimizer, restored_scheduler, _, _ = load_checkpoint(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                checkpoint_path,
                device='cpu',
                lr_monitor='val',
            )

        self.assertAlmostEqual(restored_scheduler.best, 0.5)
        self.assertEqual(restored_scheduler.num_bad_epochs, 0)
        self.assertAlmostEqual(restored_optimizer.param_groups[0]['lr'], 1.0)

    def test_marked_post_step_checkpoint_is_not_stepped_twice(self):
        model, optimizer, scheduler = _training_objects()
        scheduler.step(1.0)
        scheduler.step(2.0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'post-step.pth')
            payload = save_model_checkpoint(
                checkpoint_path,
                model,
                epoch=2,
                training_params={
                    'checkpoint': {
                        'exclude_frozen_encoder': False,
                        'save_optimizer_state': True,
                    },
                },
                optimizer=optimizer,
                scheduler=scheduler,
                loss=2.0,
                scheduler_step_completed=True,
                scheduler_monitor='val',
                scheduler_monitor_loss=2.0,
            )
            restored_model, restored_optimizer, restored_scheduler = _training_objects()
            _, restored_optimizer, restored_scheduler, _, _ = load_checkpoint(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                checkpoint_path,
                device='cpu',
                lr_monitor='val',
            )

        self.assertTrue(payload['scheduler_step_completed'])
        self.assertEqual(payload['scheduler_state_timing'], 'after_epoch_step')
        self.assertAlmostEqual(restored_optimizer.param_groups[0]['lr'], 0.5)
        self.assertEqual(restored_scheduler.num_bad_epochs, 0)


if __name__ == '__main__':
    unittest.main()
