"""
Global Multi-Region Deployment Orchestrator
Advanced production deployment with global optimization and auto-scaling.
"""

import asyncio
import time
import json
import hashlib
import random
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import concurrent.futures
from abc import ABC, abstractmethod

try:
    import aiohttp
    import asyncio
except ImportError:
    aiohttp = None
    asyncio = None

try:
    import psutil
except ImportError:
    psutil = None

from ..utils.logging import get_logger
from ..utils.metrics import record_operation_metric
from ..utils.exceptions import DeploymentError, ResourceError
from ..optimization.quantum_performance import QuantumTaskScheduler, AdaptiveResourceManager


class DeploymentRegion(Enum):
    """Global deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    AUSTRALIA = "ap-southeast-2"
    SOUTH_AMERICA = "sa-east-1"
    CANADA = "ca-central-1"
    INDIA = "ap-south-1"


class ServiceStatus(Enum):
    """Service deployment status."""
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    SCALING = "scaling"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


@dataclass
class RegionConfiguration:
    """Configuration for deployment region."""
    region: DeploymentRegion
    availability_zones: List[str]
    instance_types: List[str]
    auto_scaling_config: Dict[str, Any]
    network_config: Dict[str, Any]
    compliance_requirements: List[str]
    latency_targets: Dict[str, float]
    cost_optimization: Dict[str, Any]


@dataclass
class ServiceDeployment:
    """Service deployment information."""
    service_name: str
    region: DeploymentRegion
    status: ServiceStatus
    instances: List[Dict[str, Any]]
    load_balancer: Dict[str, Any]
    health_checks: Dict[str, Any]
    metrics: Dict[str, float]
    last_updated: float
    deployment_id: str


class GlobalLoadBalancer:
    """
    Global load balancer with intelligent traffic routing.
    
    Features:
    - Geographic routing
    - Latency-based routing
    - Health-aware routing
    - Cost-optimized routing
    """
    
    def __init__(self):
        self.logger = get_logger("GlobalLoadBalancer")
        self.region_weights = {}
        self.routing_table = {}
        self.health_monitors = {}
        self.traffic_patterns = defaultdict(lambda: deque(maxlen=1000))
        
        # Routing algorithms
        self.routing_algorithms = {
            'geographic': self._geographic_routing,
            'latency_based': self._latency_based_routing,
            'cost_optimized': self._cost_optimized_routing,
            'load_balanced': self._load_balanced_routing,
            'intelligent': self._intelligent_routing
        }
        
        # Initialize region configurations
        self.region_configs = self._initialize_region_configs()
    
    def _initialize_region_configs(self) -> Dict[DeploymentRegion, RegionConfiguration]:
        """Initialize regional configurations."""
        configs = {}
        
        # US East (Virginia)
        configs[DeploymentRegion.US_EAST] = RegionConfiguration(
            region=DeploymentRegion.US_EAST,
            availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
            instance_types=["t3.medium", "t3.large", "c5.xlarge", "m5.large"],
            auto_scaling_config={
                "min_capacity": 2,
                "max_capacity": 20,
                "target_cpu_utilization": 70,
                "scale_out_cooldown": 300,
                "scale_in_cooldown": 300
            },
            network_config={
                "vpc_cidr": "10.0.0.0/16",
                "public_subnets": ["10.0.1.0/24", "10.0.2.0/24"],
                "private_subnets": ["10.0.10.0/24", "10.0.20.0/24"]
            },
            compliance_requirements=["SOC2", "HIPAA", "PCI-DSS"],
            latency_targets={"p50": 50, "p95": 150, "p99": 300},
            cost_optimization={
                "spot_instances_enabled": True,
                "reserved_instances_ratio": 0.6,
                "cost_per_hour_target": 0.15
            }
        )
        
        # EU West (Ireland)
        configs[DeploymentRegion.EU_WEST] = RegionConfiguration(
            region=DeploymentRegion.EU_WEST,
            availability_zones=["eu-west-1a", "eu-west-1b", "eu-west-1c"],
            instance_types=["t3.medium", "t3.large", "c5.xlarge", "m5.large"],
            auto_scaling_config={
                "min_capacity": 2,
                "max_capacity": 15,
                "target_cpu_utilization": 75,
                "scale_out_cooldown": 300,
                "scale_in_cooldown": 300
            },
            network_config={
                "vpc_cidr": "10.1.0.0/16",
                "public_subnets": ["10.1.1.0/24", "10.1.2.0/24"],
                "private_subnets": ["10.1.10.0/24", "10.1.20.0/24"]
            },
            compliance_requirements=["GDPR", "ISO27001", "SOC2"],
            latency_targets={"p50": 40, "p95": 120, "p99": 250},
            cost_optimization={
                "spot_instances_enabled": True,
                "reserved_instances_ratio": 0.7,
                "cost_per_hour_target": 0.18
            }
        )
        
        # Asia Pacific (Singapore)
        configs[DeploymentRegion.ASIA_PACIFIC] = RegionConfiguration(
            region=DeploymentRegion.ASIA_PACIFIC,
            availability_zones=["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"],
            instance_types=["t3.medium", "t3.large", "c5.large", "m5.large"],
            auto_scaling_config={
                "min_capacity": 1,
                "max_capacity": 12,
                "target_cpu_utilization": 80,
                "scale_out_cooldown": 300,
                "scale_in_cooldown": 300
            },
            network_config={
                "vpc_cidr": "10.2.0.0/16",
                "public_subnets": ["10.2.1.0/24", "10.2.2.0/24"],
                "private_subnets": ["10.2.10.0/24", "10.2.20.0/24"]
            },
            compliance_requirements=["PDPA", "ISO27001"],
            latency_targets={"p50": 60, "p95": 180, "p99": 400},
            cost_optimization={
                "spot_instances_enabled": True,
                "reserved_instances_ratio": 0.5,
                "cost_per_hour_target": 0.12
            }
        )
        
        return configs
    
    def register_service_endpoint(self, service_name: str, region: DeploymentRegion, 
                                endpoint_url: str, health_check_url: str):
        """Register service endpoint for load balancing."""
        if service_name not in self.routing_table:
            self.routing_table[service_name] = {}
        
        self.routing_table[service_name][region] = {
            'endpoint_url': endpoint_url,
            'health_check_url': health_check_url,
            'status': 'active',
            'latency': 0.0,
            'success_rate': 1.0,
            'current_load': 0,
            'registered_at': time.time()
        }
        
        # Initialize health monitoring
        if service_name not in self.health_monitors:
            self.health_monitors[service_name] = {}
        
        self.health_monitors[service_name][region] = {
            'last_check': 0,
            'consecutive_failures': 0,
            'total_checks': 0,
            'successful_checks': 0
        }
        
        self.logger.info(f"Registered {service_name} endpoint in {region.value}")
    
    async def route_request(self, service_name: str, client_location: Dict[str, float], 
                          routing_algorithm: str = 'intelligent') -> Dict[str, Any]:
        """
        Route request to optimal service endpoint.
        
        Args:
            service_name: Name of service to route to
            client_location: Client geographic location {'lat': float, 'lng': float}
            routing_algorithm: Routing algorithm to use
            
        Returns:
            Routing decision with endpoint information
        """
        if service_name not in self.routing_table:
            raise DeploymentError(f"Service not found: {service_name}")
        
        available_regions = [
            region for region, endpoint in self.routing_table[service_name].items()
            if endpoint['status'] == 'active'
        ]
        
        if not available_regions:
            raise DeploymentError(f"No healthy endpoints for service: {service_name}")
        
        # Select routing algorithm
        routing_func = self.routing_algorithms.get(routing_algorithm, 
                                                  self._intelligent_routing)
        
        # Route request
        selected_region, routing_metadata = await routing_func(
            service_name, client_location, available_regions)
        
        endpoint_info = self.routing_table[service_name][selected_region]
        
        # Update traffic patterns
        self.traffic_patterns[service_name].append({
            'region': selected_region,
            'client_location': client_location,
            'timestamp': time.time(),
            'routing_algorithm': routing_algorithm
        })
        
        # Update endpoint load
        endpoint_info['current_load'] += 1
        
        result = {
            'endpoint_url': endpoint_info['endpoint_url'],
            'region': selected_region,
            'routing_metadata': routing_metadata,
            'estimated_latency': endpoint_info['latency'],
            'load_factor': endpoint_info['current_load']
        }
        
        record_operation_metric('global_load_balancer_routing', 1, {
            'service': service_name,
            'region': selected_region.value,
            'algorithm': routing_algorithm
        })
        
        return result
    
    async def _geographic_routing(self, service_name: str, client_location: Dict[str, float], 
                                available_regions: List[DeploymentRegion]) -> Tuple[DeploymentRegion, Dict]:
        """Route based on geographic proximity."""
        # Mock region coordinates
        region_coordinates = {
            DeploymentRegion.US_EAST: {'lat': 38.13, 'lng': -78.45},
            DeploymentRegion.US_WEST: {'lat': 37.42, 'lng': -122.08},
            DeploymentRegion.EU_WEST: {'lat': 53.41, 'lng': -8.24},
            DeploymentRegion.EU_CENTRAL: {'lat': 50.11, 'lng': 8.68},
            DeploymentRegion.ASIA_PACIFIC: {'lat': 1.29, 'lng': 103.85},
            DeploymentRegion.ASIA_NORTHEAST: {'lat': 35.41, 'lng': 139.42},
            DeploymentRegion.AUSTRALIA: {'lat': -33.86, 'lng': 151.20},
            DeploymentRegion.SOUTH_AMERICA: {'lat': -23.34, 'lng': -46.38},
            DeploymentRegion.CANADA: {'lat': 45.50, 'lng': -73.56},
            DeploymentRegion.INDIA: {'lat': 19.01, 'lng': 72.85}
        }
        
        min_distance = float('inf')
        closest_region = available_regions[0]
        
        for region in available_regions:
            if region in region_coordinates:
                region_coords = region_coordinates[region]
                
                # Calculate great circle distance (simplified)
                lat_diff = abs(client_location['lat'] - region_coords['lat'])
                lng_diff = abs(client_location['lng'] - region_coords['lng'])
                distance = math.sqrt(lat_diff**2 + lng_diff**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_region = region
        
        metadata = {
            'algorithm': 'geographic',
            'distance': min_distance,
            'client_location': client_location,
            'selected_coordinates': region_coordinates.get(closest_region)
        }
        
        return closest_region, metadata
    
    async def _latency_based_routing(self, service_name: str, client_location: Dict[str, float], 
                                   available_regions: List[DeploymentRegion]) -> Tuple[DeploymentRegion, Dict]:
        """Route based on measured latency."""
        region_latencies = {}
        
        for region in available_regions:
            endpoint_info = self.routing_table[service_name][region]
            # Use stored latency or estimate based on region config
            if endpoint_info['latency'] > 0:
                region_latencies[region] = endpoint_info['latency']
            else:
                # Estimate latency based on region configuration
                region_config = self.region_configs.get(region)
                if region_config:
                    estimated_latency = region_config.latency_targets['p50'] + random.uniform(-10, 20)
                    region_latencies[region] = max(10, estimated_latency)
                else:
                    region_latencies[region] = 100  # Default estimate
        
        # Select region with lowest latency
        best_region = min(region_latencies.keys(), key=lambda r: region_latencies[r])
        
        metadata = {
            'algorithm': 'latency_based',
            'region_latencies': region_latencies,
            'selected_latency': region_latencies[best_region]
        }
        
        return best_region, metadata
    
    async def _cost_optimized_routing(self, service_name: str, client_location: Dict[str, float], 
                                    available_regions: List[DeploymentRegion]) -> Tuple[DeploymentRegion, Dict]:
        """Route based on cost optimization."""
        region_costs = {}
        
        for region in available_regions:
            region_config = self.region_configs.get(region)
            if region_config:
                # Calculate cost score (lower is better)
                cost_per_hour = region_config.cost_optimization['cost_per_hour_target']
                current_load = self.routing_table[service_name][region]['current_load']
                
                # Adjust cost based on current utilization
                utilization_factor = 1.0 + (current_load * 0.1)  # 10% increase per load unit
                adjusted_cost = cost_per_hour * utilization_factor
                
                region_costs[region] = adjusted_cost
            else:
                region_costs[region] = 0.20  # Default cost
        
        # Select most cost-effective region
        best_region = min(region_costs.keys(), key=lambda r: region_costs[r])
        
        metadata = {
            'algorithm': 'cost_optimized',
            'region_costs': region_costs,
            'selected_cost': region_costs[best_region]
        }
        
        return best_region, metadata
    
    async def _load_balanced_routing(self, service_name: str, client_location: Dict[str, float], 
                                   available_regions: List[DeploymentRegion]) -> Tuple[DeploymentRegion, Dict]:
        """Route based on current load balancing."""
        region_loads = {}
        
        for region in available_regions:
            endpoint_info = self.routing_table[service_name][region]
            region_loads[region] = endpoint_info['current_load']
        
        # Select region with lowest current load
        best_region = min(region_loads.keys(), key=lambda r: region_loads[r])
        
        metadata = {
            'algorithm': 'load_balanced',
            'region_loads': region_loads,
            'selected_load': region_loads[best_region]
        }
        
        return best_region, metadata
    
    async def _intelligent_routing(self, service_name: str, client_location: Dict[str, float], 
                                 available_regions: List[DeploymentRegion]) -> Tuple[DeploymentRegion, Dict]:
        """Intelligent routing combining multiple factors."""
        # Get routing results from all algorithms
        geo_region, geo_metadata = await self._geographic_routing(
            service_name, client_location, available_regions)
        latency_region, latency_metadata = await self._latency_based_routing(
            service_name, client_location, available_regions)
        cost_region, cost_metadata = await self._cost_optimized_routing(
            service_name, client_location, available_regions)
        load_region, load_metadata = await self._load_balanced_routing(
            service_name, client_location, available_regions)
        
        # Calculate composite score for each region
        region_scores = {}
        
        for region in available_regions:
            score = 0.0
            
            # Geographic proximity score (0-1, higher is better)
            if region == geo_region:
                geo_score = 1.0
            else:
                geo_distance = geo_metadata.get('distance', 100)
                geo_score = max(0.0, 1.0 - (geo_distance / 100))
            
            # Latency score (0-1, higher is better)
            region_latency = latency_metadata['region_latencies'][region]
            min_latency = min(latency_metadata['region_latencies'].values())
            latency_score = min_latency / region_latency if region_latency > 0 else 0.5
            
            # Cost score (0-1, higher is better)
            region_cost = cost_metadata['region_costs'][region]
            min_cost = min(cost_metadata['region_costs'].values())
            cost_score = min_cost / region_cost if region_cost > 0 else 0.5
            
            # Load balancing score (0-1, higher is better)
            region_load = load_metadata['region_loads'][region]
            max_load = max(load_metadata['region_loads'].values()) + 1  # Avoid division by zero
            load_score = 1.0 - (region_load / max_load)
            
            # Health score
            health_monitor = self.health_monitors.get(service_name, {}).get(region, {})
            if health_monitor.get('total_checks', 0) > 0:
                health_score = health_monitor['successful_checks'] / health_monitor['total_checks']
            else:
                health_score = 1.0  # Assume healthy if no data
            
            # Weighted composite score
            weights = {
                'geography': 0.25,
                'latency': 0.30,
                'cost': 0.15,
                'load': 0.20,
                'health': 0.10
            }
            
            composite_score = (
                geo_score * weights['geography'] +
                latency_score * weights['latency'] +
                cost_score * weights['cost'] +
                load_score * weights['load'] +
                health_score * weights['health']
            )
            
            region_scores[region] = {
                'composite_score': composite_score,
                'geo_score': geo_score,
                'latency_score': latency_score,
                'cost_score': cost_score,
                'load_score': load_score,
                'health_score': health_score
            }
        
        # Select region with highest composite score
        best_region = max(region_scores.keys(), 
                         key=lambda r: region_scores[r]['composite_score'])
        
        metadata = {
            'algorithm': 'intelligent',
            'region_scores': region_scores,
            'selected_region': best_region,
            'selected_score': region_scores[best_region],
            'sub_algorithm_results': {
                'geographic': geo_metadata,
                'latency': latency_metadata,
                'cost': cost_metadata,
                'load': load_metadata
            }
        }
        
        return best_region, metadata
    
    async def health_check_endpoints(self):
        """Perform health checks on all registered endpoints."""
        health_check_tasks = []
        
        for service_name in self.routing_table:
            for region in self.routing_table[service_name]:
                endpoint_info = self.routing_table[service_name][region]
                health_check_url = endpoint_info['health_check_url']
                
                task = self._check_endpoint_health(service_name, region, health_check_url)
                health_check_tasks.append(task)
        
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_endpoint_health(self, service_name: str, region: DeploymentRegion, 
                                   health_check_url: str):
        """Check health of individual endpoint."""
        monitor = self.health_monitors[service_name][region]
        endpoint_info = self.routing_table[service_name][region]
        
        start_time = time.time()
        
        try:
            # Mock health check (replace with actual HTTP request in production)
            await asyncio.sleep(random.uniform(0.01, 0.1))  # Simulate network delay
            
            # Random health check result (90% success rate)
            is_healthy = random.random() < 0.9
            
            if is_healthy:
                endpoint_info['status'] = 'active'
                endpoint_info['latency'] = (time.time() - start_time) * 1000  # Convert to ms
                endpoint_info['success_rate'] = min(1.0, endpoint_info['success_rate'] * 1.01)
                
                monitor['successful_checks'] += 1
                monitor['consecutive_failures'] = 0
            else:
                monitor['consecutive_failures'] += 1
                endpoint_info['success_rate'] = max(0.0, endpoint_info['success_rate'] * 0.95)
                
                if monitor['consecutive_failures'] >= 3:
                    endpoint_info['status'] = 'unhealthy'
                    self.logger.warning(f"Endpoint unhealthy: {service_name} in {region.value}")
            
            monitor['total_checks'] += 1
            monitor['last_check'] = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed for {service_name} in {region.value}: {e}")
            monitor['consecutive_failures'] += 1
            monitor['total_checks'] += 1
            
            if monitor['consecutive_failures'] >= 5:
                endpoint_info['status'] = 'unhealthy'


class GlobalDeploymentOrchestrator:
    """
    Global deployment orchestrator with intelligent scaling and management.
    
    Features:
    - Multi-region deployment
    - Intelligent auto-scaling
    - Blue-green deployments
    - Disaster recovery
    - Cost optimization
    """
    
    def __init__(self):
        self.logger = get_logger("GlobalDeploymentOrchestrator")
        self.load_balancer = GlobalLoadBalancer()
        self.quantum_scheduler = None
        self.resource_manager = None
        
        # Deployment state
        self.active_deployments = {}
        self.deployment_history = deque(maxlen=1000)
        self.scaling_decisions = defaultdict(list)
        
        # Configuration
        self.global_config = {
            'target_regions': [
                DeploymentRegion.US_EAST,
                DeploymentRegion.EU_WEST,
                DeploymentRegion.ASIA_PACIFIC
            ],
            'disaster_recovery_regions': [
                DeploymentRegion.US_WEST,
                DeploymentRegion.EU_CENTRAL
            ],
            'deployment_strategy': 'blue_green',
            'auto_scaling_enabled': True,
            'cost_optimization_enabled': True,
            'compliance_mode': 'strict'
        }
        
        # Monitoring
        self.deployment_metrics = {
            'total_deployments': 0,
            'successful_deployments': 0,
            'failed_deployments': 0,
            'scaling_actions': 0,
            'disaster_recovery_activations': 0
        }
    
    async def initialize(self):
        """Initialize deployment orchestrator."""
        # Initialize quantum performance components
        from ..optimization.quantum_performance import create_quantum_performance_system
        self.quantum_scheduler, self.resource_manager = await create_quantum_performance_system()
        
        # Start background tasks
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._auto_scaling_loop())
        
        self.logger.info("Global deployment orchestrator initialized")
    
    async def deploy_service_globally(self, service_name: str, service_config: Dict[str, Any], 
                                    target_regions: List[DeploymentRegion] = None) -> Dict[str, Any]:
        """
        Deploy service to multiple regions globally.
        
        Args:
            service_name: Name of service to deploy
            service_config: Service configuration
            target_regions: List of regions to deploy to
            
        Returns:
            Deployment results
        """
        target_regions = target_regions or self.global_config['target_regions']
        deployment_id = f"deploy_{service_name}_{int(time.time())}"
        
        self.logger.info(f"Starting global deployment: {deployment_id}")
        
        deployment_tasks = []
        
        # Create deployment tasks for each region
        for region in target_regions:
            task = self._deploy_to_region(
                service_name, service_config, region, deployment_id)
            deployment_tasks.append(task)
        
        # Execute deployments in parallel
        deployment_results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        # Process results
        successful_deployments = []
        failed_deployments = []
        
        for i, result in enumerate(deployment_results):
            region = target_regions[i]
            
            if isinstance(result, Exception):
                failed_deployments.append({
                    'region': region,
                    'error': str(result)
                })
                self.logger.error(f"Deployment failed in {region.value}: {result}")
            else:
                successful_deployments.append(result)
                
                # Register endpoint with load balancer
                self.load_balancer.register_service_endpoint(
                    service_name=service_name,
                    region=region,
                    endpoint_url=result['endpoint_url'],
                    health_check_url=result['health_check_url']
                )
        
        # Update metrics
        self.deployment_metrics['total_deployments'] += 1
        self.deployment_metrics['successful_deployments'] += len(successful_deployments)
        self.deployment_metrics['failed_deployments'] += len(failed_deployments)
        
        # Store deployment record
        deployment_record = {
            'deployment_id': deployment_id,
            'service_name': service_name,
            'target_regions': [r.value for r in target_regions],
            'successful_deployments': successful_deployments,
            'failed_deployments': failed_deployments,
            'timestamp': time.time(),
            'status': 'completed' if successful_deployments else 'failed'
        }
        
        self.deployment_history.append(deployment_record)
        
        # Update active deployments
        if successful_deployments:
            self.active_deployments[service_name] = {
                'deployments': successful_deployments,
                'last_updated': time.time(),
                'deployment_id': deployment_id
            }
        
        result = {
            'deployment_id': deployment_id,
            'successful_regions': len(successful_deployments),
            'failed_regions': len(failed_deployments),
            'successful_deployments': successful_deployments,
            'failed_deployments': failed_deployments,
            'global_endpoint': f"https://global-lb.watermark-lab.com/{service_name}",
            'deployment_time': time.time()
        }
        
        record_operation_metric('global_service_deployment', 1, {
            'service': service_name,
            'successful_regions': len(successful_deployments),
            'failed_regions': len(failed_deployments)
        })
        
        return result
    
    async def _deploy_to_region(self, service_name: str, service_config: Dict[str, Any], 
                              region: DeploymentRegion, deployment_id: str) -> Dict[str, Any]:
        """Deploy service to specific region."""
        region_config = self.load_balancer.region_configs.get(region)
        if not region_config:
            raise DeploymentError(f"No configuration found for region: {region.value}")
        
        # Mock deployment process (replace with actual deployment logic)
        deployment_start = time.time()
        
        # Simulate deployment steps
        steps = [
            "Creating infrastructure",
            "Deploying application",
            "Configuring load balancer",
            "Running health checks",
            "Registering service"
        ]
        
        for step in steps:
            self.logger.info(f"[{region.value}] {step}...")
            await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate work
        
        # Generate deployment result
        deployment_time = time.time() - deployment_start
        
        # Mock endpoint URLs
        region_code = region.value.replace('-', '')
        endpoint_url = f"https://{service_name}-{region_code}.watermark-lab.com"
        health_check_url = f"{endpoint_url}/health"
        
        # Create service deployment record
        service_deployment = ServiceDeployment(
            service_name=service_name,
            region=region,
            status=ServiceStatus.ACTIVE,
            instances=[
                {
                    'instance_id': f"i-{hashlib.md5((service_name + region.value + str(i)).encode()).hexdigest()[:8]}",
                    'instance_type': random.choice(region_config.instance_types),
                    'availability_zone': random.choice(region_config.availability_zones),
                    'status': 'running',
                    'launch_time': time.time()
                }
                for i in range(region_config.auto_scaling_config['min_capacity'])
            ],
            load_balancer={
                'dns_name': f"{service_name}-lb-{region_code}.elb.amazonaws.com",
                'health_check_path': '/health',
                'health_check_interval': 30
            },
            health_checks={
                'enabled': True,
                'path': '/health',
                'interval': 30,
                'timeout': 5,
                'healthy_threshold': 2,
                'unhealthy_threshold': 5
            },
            metrics={
                'cpu_utilization': random.uniform(20, 40),
                'memory_utilization': random.uniform(30, 50),
                'request_count': 0,
                'error_rate': 0.0
            },
            last_updated=time.time(),
            deployment_id=deployment_id
        )
        
        result = {
            'region': region,
            'endpoint_url': endpoint_url,
            'health_check_url': health_check_url,
            'deployment_time': deployment_time,
            'instances': len(service_deployment.instances),
            'service_deployment': service_deployment
        }
        
        return result
    
    async def scale_service(self, service_name: str, region: DeploymentRegion, 
                          target_capacity: int, reason: str = "manual") -> Dict[str, Any]:
        """
        Scale service in specific region.
        
        Args:
            service_name: Service to scale
            region: Target region
            target_capacity: Desired instance count
            reason: Scaling reason
            
        Returns:
            Scaling operation result
        """
        if service_name not in self.active_deployments:
            raise DeploymentError(f"Service not found: {service_name}")
        
        # Find deployment in region
        target_deployment = None
        for deployment in self.active_deployments[service_name]['deployments']:
            if deployment['service_deployment'].region == region:
                target_deployment = deployment['service_deployment']
                break
        
        if not target_deployment:
            raise DeploymentError(f"Service {service_name} not deployed in {region.value}")
        
        current_capacity = len(target_deployment.instances)
        region_config = self.load_balancer.region_configs[region]
        
        # Validate target capacity
        min_capacity = region_config.auto_scaling_config['min_capacity']
        max_capacity = region_config.auto_scaling_config['max_capacity']
        
        if target_capacity < min_capacity:
            target_capacity = min_capacity
        elif target_capacity > max_capacity:
            target_capacity = max_capacity
        
        if target_capacity == current_capacity:
            return {
                'action': 'no_change',
                'current_capacity': current_capacity,
                'target_capacity': target_capacity,
                'reason': 'capacity_already_at_target'
            }
        
        # Perform scaling operation
        scaling_start = time.time()
        action = 'scale_out' if target_capacity > current_capacity else 'scale_in'
        
        self.logger.info(f"Scaling {service_name} in {region.value}: {current_capacity} -> {target_capacity} ({reason})")
        
        if action == 'scale_out':
            # Add instances
            new_instances = []
            for i in range(target_capacity - current_capacity):
                instance_id = f"i-{hashlib.md5((service_name + region.value + str(time.time()) + str(i)).encode()).hexdigest()[:8]}"
                new_instance = {
                    'instance_id': instance_id,
                    'instance_type': random.choice(region_config.instance_types),
                    'availability_zone': random.choice(region_config.availability_zones),
                    'status': 'pending',
                    'launch_time': time.time()
                }
                new_instances.append(new_instance)
            
            # Simulate instance launch time
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Update instance status
            for instance in new_instances:
                instance['status'] = 'running'
            
            target_deployment.instances.extend(new_instances)
            
        else:  # scale_in
            # Remove instances
            instances_to_remove = current_capacity - target_capacity
            removed_instances = target_deployment.instances[-instances_to_remove:]
            target_deployment.instances = target_deployment.instances[:-instances_to_remove]
            
            # Simulate instance termination time
            await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Update deployment status
        target_deployment.status = ServiceStatus.ACTIVE
        target_deployment.last_updated = time.time()
        
        scaling_time = time.time() - scaling_start
        
        # Record scaling decision
        scaling_record = {
            'service_name': service_name,
            'region': region.value,
            'action': action,
            'previous_capacity': current_capacity,
            'new_capacity': target_capacity,
            'reason': reason,
            'scaling_time': scaling_time,
            'timestamp': time.time()
        }
        
        self.scaling_decisions[service_name].append(scaling_record)
        self.deployment_metrics['scaling_actions'] += 1
        
        result = {
            'action': action,
            'previous_capacity': current_capacity,
            'new_capacity': len(target_deployment.instances),
            'scaling_time': scaling_time,
            'reason': reason,
            'instances_changed': abs(target_capacity - current_capacity)
        }
        
        record_operation_metric('service_scaling', 1, {
            'service': service_name,
            'region': region.value,
            'action': action
        })
        
        return result
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Perform health checks
                await self.load_balancer.health_check_endpoints()
                
                # Update deployment metrics
                await self._update_deployment_metrics()
                
                # Check for anomalies
                await self._check_deployment_anomalies()
                
                # Sleep before next iteration
                await asyncio.sleep(30)  # 30-second monitoring interval
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _auto_scaling_loop(self):
        """Background auto-scaling loop."""
        while True:
            try:
                if self.global_config['auto_scaling_enabled']:
                    await self._perform_auto_scaling_analysis()
                
                await asyncio.sleep(60)  # 60-second scaling check interval
                
            except Exception as e:
                self.logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _perform_auto_scaling_analysis(self):
        """Perform auto-scaling analysis and actions."""
        for service_name in self.active_deployments:
            service_deployments = self.active_deployments[service_name]['deployments']
            
            for deployment in service_deployments:
                service_deployment = deployment['service_deployment']
                region = service_deployment.region
                region_config = self.load_balancer.region_configs[region]
                
                # Get current metrics
                current_cpu = service_deployment.metrics['cpu_utilization']
                current_memory = service_deployment.metrics['memory_utilization']
                current_capacity = len(service_deployment.instances)
                
                # Auto-scaling thresholds
                target_cpu = region_config.auto_scaling_config['target_cpu_utilization']
                scale_out_threshold = target_cpu + 10  # Scale out at +10%
                scale_in_threshold = target_cpu - 20   # Scale in at -20%
                
                scaling_needed = False
                target_capacity = current_capacity
                reason = ""
                
                # Scale out decision
                if current_cpu > scale_out_threshold and current_capacity < region_config.auto_scaling_config['max_capacity']:
                    target_capacity = min(current_capacity + 1, region_config.auto_scaling_config['max_capacity'])
                    reason = f"High CPU utilization: {current_cpu:.1f}% > {scale_out_threshold}%"
                    scaling_needed = True
                
                # Scale in decision
                elif current_cpu < scale_in_threshold and current_capacity > region_config.auto_scaling_config['min_capacity']:
                    target_capacity = max(current_capacity - 1, region_config.auto_scaling_config['min_capacity'])
                    reason = f"Low CPU utilization: {current_cpu:.1f}% < {scale_in_threshold}%"
                    scaling_needed = True
                
                # Perform scaling if needed
                if scaling_needed and target_capacity != current_capacity:
                    try:
                        await self.scale_service(service_name, region, target_capacity, reason)
                    except Exception as e:
                        self.logger.error(f"Auto-scaling failed for {service_name} in {region.value}: {e}")
    
    async def _update_deployment_metrics(self):
        """Update deployment metrics."""
        # Mock metric updates (replace with actual monitoring integration)
        for service_name in self.active_deployments:
            service_deployments = self.active_deployments[service_name]['deployments']
            
            for deployment in service_deployments:
                service_deployment = deployment['service_deployment']
                
                # Simulate metric updates
                service_deployment.metrics.update({
                    'cpu_utilization': max(10, min(90, service_deployment.metrics['cpu_utilization'] + random.uniform(-5, 5))),
                    'memory_utilization': max(20, min(85, service_deployment.metrics['memory_utilization'] + random.uniform(-3, 3))),
                    'request_count': service_deployment.metrics['request_count'] + random.randint(0, 100),
                    'error_rate': max(0, min(10, service_deployment.metrics['error_rate'] + random.uniform(-0.5, 0.5)))
                })
    
    async def _check_deployment_anomalies(self):
        """Check for deployment anomalies and issues."""
        for service_name in self.active_deployments:
            service_deployments = self.active_deployments[service_name]['deployments']
            
            for deployment in service_deployments:
                service_deployment = deployment['service_deployment']
                
                # Check for high error rates
                if service_deployment.metrics['error_rate'] > 5.0:
                    self.logger.warning(f"High error rate detected for {service_name} in {service_deployment.region.value}: {service_deployment.metrics['error_rate']:.2f}%")
                
                # Check for resource exhaustion
                if service_deployment.metrics['cpu_utilization'] > 95:
                    self.logger.critical(f"Critical CPU utilization for {service_name} in {service_deployment.region.value}: {service_deployment.metrics['cpu_utilization']:.1f}%")
                
                if service_deployment.metrics['memory_utilization'] > 90:
                    self.logger.critical(f"Critical memory utilization for {service_name} in {service_deployment.region.value}: {service_deployment.metrics['memory_utilization']:.1f}%")
    
    def get_global_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive global deployment status."""
        status = {
            'active_services': len(self.active_deployments),
            'total_regions': len(set(dep['service_deployment'].region 
                                    for service in self.active_deployments.values()
                                    for dep in service['deployments'])),
            'total_instances': sum(len(dep['service_deployment'].instances)
                                  for service in self.active_deployments.values()
                                  for dep in service['deployments']),
            'deployment_metrics': self.deployment_metrics.copy(),
            'services': {}
        }
        
        # Service-specific status
        for service_name in self.active_deployments:
            service_data = self.active_deployments[service_name]
            deployments_info = []
            
            for deployment in service_data['deployments']:
                service_deployment = deployment['service_deployment']
                deployments_info.append({
                    'region': service_deployment.region.value,
                    'status': service_deployment.status.value,
                    'instances': len(service_deployment.instances),
                    'metrics': service_deployment.metrics,
                    'last_updated': service_deployment.last_updated
                })
            
            status['services'][service_name] = {
                'deployments': deployments_info,
                'total_instances': sum(len(dep['service_deployment'].instances) 
                                     for dep in service_data['deployments']),
                'regions': [dep['service_deployment'].region.value 
                           for dep in service_data['deployments']],
                'last_updated': service_data['last_updated']
            }
        
        return status


# Integration function
async def create_global_deployment_system() -> GlobalDeploymentOrchestrator:
    """Create and initialize global deployment system."""
    orchestrator = GlobalDeploymentOrchestrator()
    await orchestrator.initialize()
    return orchestrator


# Example usage and testing
async def test_global_deployment():
    """Test global deployment system."""
    print("\n🌍 Global Deployment System Test")
    print("=" * 40)
    
    # Initialize system
    orchestrator = await create_global_deployment_system()
    
    # Test service configuration
    watermark_service_config = {
        'image': 'watermark-lab:latest',
        'cpu': '500m',
        'memory': '1Gi',
        'min_replicas': 2,
        'max_replicas': 10,
        'environment': {
            'LOG_LEVEL': 'INFO',
            'METRICS_ENABLED': 'true'
        }
    }
    
    # Deploy watermark service globally
    print("\n🚀 Deploying watermark service globally...")
    
    deployment_result = await orchestrator.deploy_service_globally(
        service_name="watermark-api",
        service_config=watermark_service_config
    )
    
    print(f"  Deployment ID: {deployment_result['deployment_id']}")
    print(f"  Successful regions: {deployment_result['successful_regions']}")
    print(f"  Failed regions: {deployment_result['failed_regions']}")
    print(f"  Global endpoint: {deployment_result['global_endpoint']}")
    
    # Test load balancing
    print(f"\n⚖️ Testing global load balancing...")
    
    # Simulate requests from different locations
    test_locations = [
        {'lat': 40.7128, 'lng': -74.0060},  # New York
        {'lat': 51.5074, 'lng': -0.1278},   # London  
        {'lat': 1.3521, 'lng': 103.8198},   # Singapore
        {'lat': -33.8688, 'lng': 151.2093}  # Sydney
    ]
    
    for i, location in enumerate(test_locations):
        routing_result = await orchestrator.load_balancer.route_request(
            service_name="watermark-api",
            client_location=location,
            routing_algorithm="intelligent"
        )
        
        location_name = ["New York", "London", "Singapore", "Sydney"][i]
        print(f"  {location_name}: routed to {routing_result['region'].value} "
              f"(latency: {routing_result['estimated_latency']:.1f}ms)")
    
    # Test auto-scaling
    print(f"\n📈 Testing auto-scaling...")
    
    # Simulate high CPU load
    service_deployments = orchestrator.active_deployments["watermark-api"]['deployments']
    for deployment in service_deployments:
        service_deployment = deployment['service_deployment']
        service_deployment.metrics['cpu_utilization'] = 85.0  # High CPU
    
    # Wait for auto-scaling analysis
    await orchestrator._perform_auto_scaling_analysis()
    
    # Get deployment status
    print(f"\n📊 Global deployment status:")
    status = orchestrator.get_global_deployment_status()
    
    print(f"  Active services: {status['active_services']}")
    print(f"  Total regions: {status['total_regions']}")  
    print(f"  Total instances: {status['total_instances']}")
    print(f"  Successful deployments: {status['deployment_metrics']['successful_deployments']}")
    print(f"  Scaling actions: {status['deployment_metrics']['scaling_actions']}")
    
    print(f"\n✅ Global deployment system test completed!")


if __name__ == "__main__":
    asyncio.run(test_global_deployment())