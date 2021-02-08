import yaml
from unittest import TestCase

from train_on_docker import restamp_hypertune

class Test(TestCase):
    def test_restamp_hypertune(self):
        with open("test_config.yaml", 'r') as stream:
            data = yaml.safe_load(stream)
        self.assertEqual(data["behaviors"]["RollerBall"]["hyperparameters"]["buffer_size"], 100)
        self.assertEqual(data["behaviors"]["RollerBall"]["hyperparameters"]["learning_rate_schedule"], "linear")
        self.assertEqual(data["behaviors"]["RollerBall"]["reward_signals"]["extrinsic"]["gamma"], 0.99)
        self.assertFalse(data["behaviors"]["RollerBall"]["network_settings"]["normalize"])

        arguments = ['--behaviors-RollerBall-hyperparameters-beta=0.5',
                     "--behaviors-RollerBall-hyperparameters-buffer_size=99",
                     "--behaviors-RollerBall-hyperparameters-learning_rate_schedule=none",
                     "--behaviors-RollerBall-reward_signals-extrinsic-gamma=1.99",
                     "--behaviors-RollerBall-network_settings-normalize=true",
                     '--behaviors-RollerBall-network_settings-hidden_units=196',
                     '--behaviors-RollerBall-hyperparameters-learning_rate=0.00034344872808012678']

        restamp_hypertune("test_config.yaml", arguments, "test_config_dest.yaml")
        with open("test_config_dest.yaml", 'r') as stream:
            data = yaml.safe_load(stream)
        self.assertEqual(data["behaviors"]["RollerBall"]["hyperparameters"]["buffer_size"], 99)
        self.assertEqual(data["behaviors"]["RollerBall"]["hyperparameters"]["learning_rate_schedule"], "none")
        self.assertEqual(data["behaviors"]["RollerBall"]["reward_signals"]["extrinsic"]["gamma"], 1.99)
        self.assertTrue(data["behaviors"]["RollerBall"]["network_settings"]["normalize"])
