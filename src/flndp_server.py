import threading, time, torch, yaml

import flwr as fl

from collections import OrderedDict
from flndp_model import BreastCancerMLP
from typing import Optional

from flwr.server.server_config import ServerConfig
from flwr.server.strategy import FedAvg

class FLServer:
    def __init__(self, config_path: str):
        """Initialise the FedAvg server with a config file"""
        self.config_path = config_path
        self.config = self._load_config()
        self.global_model = self._create_global_model()
        self.strategy = self._create_strategy()

    def _load_config(self):
        """Load the configuration from a YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            raise
    
    def _create_strategy(self):
        """Create the FedAvg strategy dynamically from the YAML configuration."""
        try:
            strat_cfg = self.config.get("strategy", {})
            return FedAvg(
                fraction_fit=strat_cfg.get("fraction_fit", 0.5),
                fraction_evaluate=strat_cfg.get("fraction_evaluate", 0.5),
                min_fit_clients=strat_cfg.get("min_fit_clients", 2),
                min_evaluate_clients=strat_cfg.get("min_evaluate_clients", 2),
                min_available_clients=strat_cfg.get("min_available_clients", 2),
                evaluate_fn=strat_cfg.get("eval_fn", None)
            )
        except Exception as e:
            print(f"Error creating strategy: {str(e)}")
            raise

    def _create_global_model(self):
        """Create the global model dynamically from the YAML configuration."""
        try:
            # Load model configuration from YAML
            model_cfg = self.config.get("model", {})
            if not model_cfg:
                raise ValueError("No 'model' section found in the config file.")

            # Instantiate the model with the architecture dictionary
            model = BreastCancerMLP(arch_config=model_cfg)
            print(f"Global model created with architecture: {model_cfg}")
            return model
        except Exception as e:
            print(f"Error creating global model: {str(e)}")
            raise

    def get_global_parameters(self):
        """Return the server's global model parameters as numpy arrays."""
        try:
            return [val.cpu().numpy() for val in self.global_model.state_dict().values()]
        except Exception as e:
            print(f"Error getting global parameters from the server: {str(e)}")
            raise

    def set_global_parameters(self, parameters):
        """Update the server's global model parameters from numpy arrays."""
        try:
            params_dict = zip(self.global_model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.global_model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Error setting global parameters of the server: {str(e)}")
            raise

    def _get_strategy(self, export_yaml_path: Optional[str] = None):
        """Create FedAvg strategy from config and optionally export it to YAML."""
        try:
            strat_cfg = self.config.get("strategy", {})

            strategy = fl.server.strategy.FedAvg(
                fraction_fit=strat_cfg.get("fraction_fit", 0.5),
                fraction_evaluate=strat_cfg.get("fraction_evaluate", 0.5),
                min_fit_clients=strat_cfg.get("min_fit_clients", 2),
                min_evaluate_clients=strat_cfg.get("min_evaluate_clients", 2),
                min_available_clients=strat_cfg.get("min_available_clients", 2),
                initial_parameters=fl.common.ndarrays_to_parameters(self.get_global_parameters())
            )

            # Export strategy config to YAML if path is provided
            if export_yaml_path is not None:
                with open(export_yaml_path, "w") as f:
                    yaml.safe_dump(strat_cfg, f)
                print(f"Strategy exported to {export_yaml_path}")

            return strategy

        except Exception as e:
            print(f"Error creating strategy: {str(e)}")
            raise

    # ------------------------------- SERVER THREAD ------------------------------ #
    def _run_server(self):
        server_cfg = self.config.get("server", {})
        address = server_cfg.get("address", "localhost:8080")
        num_rounds = server_cfg.get("num_rounds", 20)

        print("=== Starting Flower server ===")
        fl.server.start_server(
            server_address=address,
            config=ServerConfig(num_rounds=num_rounds),
            strategy=self._get_strategy()
        )
        print("=== Flower server stopped ===")

    # ------------------ THREAD BLOCKING EXECUTION OF THE SERVER ----------------- #
    def _start(self):
        """Start the Flower server asynchronously."""
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        time.sleep(1)
        print("Server is running in the background" if self._server_thread.is_alive() else "Server failed to start")

    def wait_until_finished(self):
        """Block until the learning session completes."""
        if self._server_thread:
            self._server_thread.join()
            print("Learning session completed.")

    # --------- WRAPPER FOR BLOCKING EXECUTION OF THE SERVER (CALL THIS) --------- #
    def run_server_blocking(self):
        """Start the server and wait until it finishes."""
        self._start()
        self.wait_until_finished()
