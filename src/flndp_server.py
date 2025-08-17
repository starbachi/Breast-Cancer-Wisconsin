import flwr as fl
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar, FitIns, EvaluateIns
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
import threading
import time
from collections import defaultdict
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedLearningStrategy(Strategy):
    """Federated Learning Strategy without Differential Privacy"""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        aggregation_method: str = "fedavg"
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.aggregation_method = aggregation_method
        
        # Training session management
        self.current_round = 0
        self.training_active = False
        self.global_model_weights = None
        self.client_updates = defaultdict(list)
        self.update_lock = threading.Lock()
        
        logger.info("Federated Learning Strategy initialized without Differential Privacy")

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        logger.info("Initializing global model parameters")
        return self.initial_parameters

    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Training Starter - Configure training session for clients"""
        logger.info(f"Starting training session for round {server_round}")
        
        self.current_round = server_round
        self.training_active = True
        
        # Sample clients for training
        sample_size = max(
            int(self.fraction_fit * client_manager.num_available()), 
            self.min_fit_clients
        )
        clients = client_manager.sample(sample_size)
        
        # Client Parameters - Configure parameters for each client
        config = self._get_client_parameters(server_round)
        
        logger.info(f"Selected {len(clients)} clients for training round {server_round}")
        return [(client, FitIns(parameters, config)) for client in clients]

    def _get_client_parameters(self, server_round: int) -> Dict[str, Scalar]:
        """Client Parameters - Pass latest global model and training parameters"""
        config = {
            "server_round": server_round,
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "model_version": f"global_v{server_round}",
        }
        
        logger.info(f"Client parameters configured for round {server_round}: {config}")
        return config

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Weight Aggregator - Asynchronously aggregate client updates"""
        logger.info(f"Aggregating weights from {len(results)} clients in round {server_round}")
        
        if not results:
            logger.warning("No client results received for aggregation")
            return None, {}
        
        # Log any failures
        if failures:
            logger.warning(f"Training failures: {len(failures)} clients failed")
        
        # Perform weight aggregation
        aggregated_parameters = self._perform_weight_aggregation(results)
        
        # Calculate aggregation metrics
        metrics = self._calculate_aggregation_metrics(results)
        
        logger.info(f"Weight aggregation completed for round {server_round}")
        return aggregated_parameters, metrics

    def _perform_weight_aggregation(
        self, 
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Optional[Parameters]:
        """Core weight aggregation logic"""
        if self.aggregation_method == "fedavg":
            return self._federated_averaging(results)
        else:
            logger.error(f"Unknown aggregation method: {self.aggregation_method}")
            return None

    def _federated_averaging(
        self, 
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Optional[Parameters]:
        """FedAvg aggregation algorithm"""
        try:
            # Extract weights and number of examples from each client
            weights_results = [
                (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            # Calculate total number of examples
            total_examples = sum([num_examples for _, num_examples in weights_results])
            
            # Perform weighted averaging
            aggregated_weights = []
            for i in range(len(weights_results[0][0])):  # For each layer
                layer_weights = []
                layer_total_examples = 0
                
                for weights, num_examples in weights_results:
                    layer_weights.append(weights[i] * num_examples)
                    layer_total_examples += num_examples
                
                # Average the layer weights
                aggregated_layer = sum(layer_weights) / layer_total_examples
                aggregated_weights.append(aggregated_layer)
            
            logger.info(f"FedAvg aggregation completed with {total_examples} total examples")
            return fl.common.ndarrays_to_parameters(aggregated_weights)
            
        except Exception as e:
            logger.error(f"Error in federated averaging: {e}")
            return None

    def _calculate_aggregation_metrics(
        self, 
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict[str, Scalar]:
        """Calculate metrics from aggregation process"""
        total_examples = sum([fit_res.num_examples for _, fit_res in results])
        
        # Extract loss values and ensure they are numeric
        loss_values = []
        for _, fit_res in results:
            loss_val = fit_res.metrics.get("loss", 0.0)
            if isinstance(loss_val, (int, float)):
                loss_values.append(float(loss_val))
        
        avg_loss = np.mean(loss_values) if loss_values else 0.0
        
        metrics = {
            "aggregated_clients": len(results),
            "total_examples": total_examples,
            "average_loss": avg_loss,
            "aggregation_round": self.current_round
        }
        
        return metrics

    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation for clients"""
        if not parameters:
            return []
        
        sample_size = max(
            int(self.fraction_evaluate * client_manager.num_available()),
            self.min_evaluate_clients
        )
        clients = client_manager.sample(sample_size)
        
        config: Dict[str, Scalar] = {"server_round": server_round}
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}
        
        # Extract accuracy values and ensure they are numeric
        accuracies = []
        examples = []
        for _, r in results:
            acc_val = r.metrics["accuracy"]
            if isinstance(acc_val, (int, float)):
                accuracies.append(float(acc_val) * r.num_examples)
                examples.append(r.num_examples)
        
        aggregated_accuracy = sum(accuracies) / sum(examples) if examples else 0.0
        
        metrics = {
            "accuracy": aggregated_accuracy,
            "evaluated_clients": len(results)
        }
        
        logger.info(f"Evaluation aggregation completed: accuracy={aggregated_accuracy:.4f}")
        return aggregated_accuracy, metrics

    def end_training_session(self) -> Dict[str, Any]:
        """Training Session Ender - Clean up and finalize training session"""
        logger.info(f"Ending training session for round {self.current_round}")
        
        self.training_active = False
        
        # Collect session statistics
        session_stats = {
            "round": self.current_round,
            "training_completed": True,
            "timestamp": time.time(),
            "session_duration": None  # Could be calculated if start time was tracked
        }
        
        # Clear temporary data
        with self.update_lock:
            self.client_updates.clear()
        
        logger.info("Training session ended successfully")
        return session_stats

    def update_global_model(self, new_parameters: Parameters) -> bool:
        """Global Model Updater - Update global model after training session"""
        try:
            with self.update_lock:
                self.global_model_weights = new_parameters
            
            logger.info("Global model updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
            return False

    def get_global_model(self) -> Optional[Parameters]:
        """Get current global model parameters"""
        return self.global_model_weights

    def is_training_active(self) -> bool:
        """Check if training session is currently active"""
        return self.training_active

    def evaluate(
        self, 
        server_round: int, 
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (optional - can return None to skip)"""
        # This method can be used for centralized evaluation on server
        # For now, we skip server-side evaluation and rely on federated evaluation
        return None


class FederatedLearningServer:
    """Main Federated Learning Server class"""
    
    def __init__(
        self, 
        strategy: FederatedLearningStrategy,
        server_address: str = "[::]:8080"
    ):
        self.strategy = strategy
        self.server_address = server_address
        self.server = None
        
    def start_server(
        self, 
        config: fl.server.ServerConfig = fl.server.ServerConfig(num_rounds=10)
    ) -> None:
        """Start the federated learning server"""
        logger.info(f"Starting Federated Learning Server on {self.server_address}")
        
        try:
            # Start Flower server
            fl.server.start_server(
                server_address=self.server_address,
                config=config,
                strategy=self.strategy
            )
            
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            raise

    def stop_server(self) -> None:
        """Stop the federated learning server"""
        logger.info("Stopping Federated Learning Server")
        # Server cleanup logic would go here
        

def create_federated_server(
    initial_model_path: Optional[str] = None,
    server_config: Optional[Dict] = None
) -> FederatedLearningServer:
    """Factory function to create a configured federated learning server"""
    
    # Default configuration
    default_config = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2,
        "aggregation_method": "fedavg"
    }
    
    if server_config:
        default_config.update(server_config)
    
    # Load initial parameters if provided
    initial_parameters = None
    if initial_model_path:
        try:
            with open(initial_model_path, 'rb') as f:
                initial_parameters = pickle.load(f)
                logger.info(f"Loaded initial model from {initial_model_path}")
        except Exception as e:
            logger.warning(f"Could not load initial model: {e}")
    
    # Create strategy
    strategy = FederatedLearningStrategy(
        initial_parameters=initial_parameters,
        **default_config
    )
    
    # Create server
    server = FederatedLearningServer(strategy)
    
    logger.info("Federated Learning Server created successfully")
    return server


if __name__ == "__main__":
    # Example usage
    server_config = {
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "fraction_fit": 0.8,
        "fraction_evaluate": 0.8
    }
    
    # Create and start server
    fl_server = create_federated_server(server_config=server_config)
    
    # Configure training rounds
    config = fl.server.ServerConfig(num_rounds=5)
    
    # Start the server (this will block)
    fl_server.start_server(config)