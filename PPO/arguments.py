import pprint

MODEL_UPDATE_CONFIG = "model_update"

def print_configs():
    print(f"[{MODEL_UPDATE_CONFIG} ")

def get_config(args):
    if args.config_name == MODEL_UPDATE_CONFIG:
        config = ModelUpdateConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))

    config.set_logdir(args.logdir)
    config.set_seed(args.seed)
    config.set_strategy(args.strategy)
    return config


class Config(object):
    def __init__(self):
        self.logdir = "log"
        self.seed = 0
        self.n_episodes = 50 #50
        self.n_seed_episodes = 2
        self.max_ep_len = 1000
        self.record_every = None
        self.coverage = False

        self.env_name = None
        self.action_repeat = 3
        #self.action_noise = None

        self.ensemble_size = 1 #10
        self.hidden_size = 200

        self.n_train_epochs = 10# 100
        self.batch_size = 2 #50
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.grad_clip_norm = 1000

        self.plan_horizon = 3 #30
        self.optimisation_iters = 2 #5
        self.n_candidates = 50 #500
        self.top_candidates = 5 #50

        self.expl_strategy = "information"
        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False

        self.expl_scale = 1.0
        self.reward_scale = 1.0

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_seed(self, seed):
        self.seed = seed

    def set_strategy(self, strategy):
        self.strategy = strategy

    def __repr__(self):
        return pprint.pformat(vars(self))



class ModelUpdateConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "model_update"


        self.n_train_epochs = 10 #100
        self.n_seed_episodes = 2
        self.max_ep_len = 1000
        self.expl_scale = 1.0

        self.ensemble_size = 5 #25
        self.record_every = None
        self.n_episodes = 50 #50
        #self.action_noise = None


