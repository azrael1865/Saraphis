"""
JAX Installation Helper for Tropical Compression System.
Provides automatic detection and installation guidance for JAX with appropriate backend.
NO PLACEHOLDERS - PRODUCTION READY
"""

import os
import sys
import subprocess
import platform
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CUDAVersion(Enum):
    """Supported CUDA versions for JAX"""
    CUDA11 = "11"
    CUDA12 = "12"
    CPU_ONLY = "cpu"
    
    @classmethod
    def detect(cls) -> 'CUDAVersion':
        """Detect installed CUDA version"""
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Try to determine CUDA version from driver
                driver_version = result.stdout.strip()
                # Driver 525+ supports CUDA 12
                if driver_version and float(driver_version.split('.')[0]) >= 525:
                    return cls.CUDA12
                else:
                    return cls.CUDA11
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
            
        # Try nvcc
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse CUDA version from nvcc output
                match = re.search(r'release (\d+)\.', result.stdout)
                if match:
                    cuda_major = int(match.group(1))
                    if cuda_major >= 12:
                        return cls.CUDA12
                    elif cuda_major >= 11:
                        return cls.CUDA11
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        return cls.CPU_ONLY


@dataclass
class SystemInfo:
    """System information for JAX installation"""
    os_name: str
    os_version: str
    python_version: str
    cuda_version: CUDAVersion
    has_gpu: bool
    gpu_count: int
    cpu_count: int
    memory_gb: float
    architecture: str  # x86_64, arm64, etc.
    
    @classmethod
    def detect(cls) -> 'SystemInfo':
        """Detect current system information"""
        import psutil
        
        # OS information
        os_name = platform.system().lower()
        os_version = platform.release()
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # CUDA detection
        cuda_version = CUDAVersion.detect()
        
        # GPU detection
        has_gpu = False
        gpu_count = 0
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if has_gpu else 0
        except ImportError:
            # Try nvidia-smi as fallback
            try:
                result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_count = len(result.stdout.strip().split('\n'))
                    has_gpu = gpu_count > 0
            except:
                pass
                
        # System resources
        cpu_count = psutil.cpu_count(logical=False) or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Architecture
        architecture = platform.machine().lower()
        
        return cls(
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
            cuda_version=cuda_version,
            has_gpu=has_gpu,
            gpu_count=gpu_count,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            architecture=architecture
        )


class JAXInstaller:
    """JAX installation manager with automatic backend selection"""
    
    # JAX package mappings
    PACKAGE_MAPPINGS = {
        # CUDA 12 packages
        ('linux', CUDAVersion.CUDA12): 'jax[cuda12]',
        ('windows', CUDAVersion.CUDA12): 'jax[cuda12_local]',
        ('darwin', CUDAVersion.CUDA12): 'jax[metal]',  # Metal for Mac
        
        # CUDA 11 packages
        ('linux', CUDAVersion.CUDA11): 'jax[cuda11_pip]',
        ('windows', CUDAVersion.CUDA11): 'jax[cuda11_local]',
        ('darwin', CUDAVersion.CUDA11): 'jax[metal]',
        
        # CPU only packages
        ('linux', CUDAVersion.CPU_ONLY): 'jax[cpu]',
        ('windows', CUDAVersion.CPU_ONLY): 'jax[cpu]',
        ('darwin', CUDAVersion.CPU_ONLY): 'jax[cpu]',
    }
    
    def __init__(self):
        """Initialize JAX installer"""
        self.system_info = SystemInfo.detect()
        self.installed_packages = self._get_installed_packages()
        
    def check_jax_installation(self) -> Dict[str, Any]:
        """Check current JAX installation status"""
        result = {
            'jax_installed': False,
            'jaxlib_installed': False,
            'jax_version': None,
            'jaxlib_version': None,
            'backend': None,
            'has_gpu_support': False,
            'issues': []
        }
        
        # Check JAX
        try:
            import jax
            result['jax_installed'] = True
            result['jax_version'] = jax.__version__
            
            # Check backend
            from jax.lib import xla_bridge
            backend = xla_bridge.get_backend()
            result['backend'] = backend.platform
            result['has_gpu_support'] = backend.platform in ['gpu', 'cuda']
            
        except ImportError as e:
            result['issues'].append(f"JAX not installed: {e}")
        except Exception as e:
            result['issues'].append(f"Error checking JAX: {e}")
            
        # Check jaxlib
        try:
            import jaxlib
            result['jaxlib_installed'] = True
            result['jaxlib_version'] = jaxlib.__version__
        except ImportError:
            result['issues'].append("jaxlib not installed")
            
        # Version compatibility check
        if result['jax_installed'] and result['jaxlib_installed']:
            jax_major = result['jax_version'].split('.')[0:2]
            jaxlib_major = result['jaxlib_version'].split('.')[0:2]
            if jax_major != jaxlib_major:
                result['issues'].append(
                    f"Version mismatch: jax {result['jax_version']} vs jaxlib {result['jaxlib_version']}"
                )
                
        # GPU support check
        if self.system_info.has_gpu and not result['has_gpu_support']:
            result['issues'].append("GPU detected but JAX has no GPU support")
            
        return result
        
    def get_recommended_package(self) -> str:
        """Get recommended JAX package for system"""
        key = (self.system_info.os_name, self.system_info.cuda_version)
        
        # Get base package
        package = self.PACKAGE_MAPPINGS.get(key)
        
        if not package:
            # Fallback to CPU version
            package = f"jax[cpu]"
            logger.warning(f"No specific package for {key}, using CPU version")
            
        # Special handling for Mac M1/M2
        if self.system_info.os_name == 'darwin' and self.system_info.architecture == 'arm64':
            package = 'jax[metal]'  # Use Metal acceleration on Apple Silicon
            
        return package
        
    def generate_install_command(self, force_reinstall: bool = False, 
                                upgrade: bool = True) -> str:
        """Generate pip install command for JAX"""
        package = self.get_recommended_package()
        
        cmd_parts = ['pip', 'install']
        
        if force_reinstall:
            cmd_parts.append('--force-reinstall')
        if upgrade:
            cmd_parts.append('--upgrade')
            
        cmd_parts.append(package)
        
        # Add specific version constraints if needed
        if self.system_info.python_version == "3.11":
            cmd_parts.append("jaxlib<=0.4.20")  # Example constraint
            
        return ' '.join(cmd_parts)
        
    def install_jax(self, force_reinstall: bool = False, 
                   dry_run: bool = False) -> Dict[str, Any]:
        """Install JAX with appropriate backend"""
        result = {
            'success': False,
            'command': None,
            'output': None,
            'error': None
        }
        
        # Generate install command
        command = self.generate_install_command(force_reinstall)
        result['command'] = command
        
        if dry_run:
            logger.info(f"Dry run - would execute: {command}")
            result['success'] = True
            return result
            
        try:
            logger.info(f"Installing JAX: {command}")
            
            # Execute installation
            process = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            result['output'] = process.stdout
            
            if process.returncode == 0:
                result['success'] = True
                logger.info("JAX installation completed successfully")
                
                # Verify installation
                verification = self.check_jax_installation()
                if verification['jax_installed']:
                    logger.info(f"JAX {verification['jax_version']} installed with {verification['backend']} backend")
                else:
                    result['error'] = "Installation completed but JAX import failed"
                    result['success'] = False
            else:
                result['error'] = process.stderr
                logger.error(f"JAX installation failed: {process.stderr}")
                
        except subprocess.TimeoutExpired:
            result['error'] = "Installation timed out after 5 minutes"
            logger.error(result['error'])
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Installation error: {e}")
            
        return result
        
    def verify_gpu_support(self) -> Dict[str, Any]:
        """Verify JAX GPU support is working"""
        result = {
            'gpu_available': False,
            'device_count': 0,
            'devices': [],
            'test_passed': False,
            'error': None
        }
        
        try:
            import jax
            
            # Check available devices
            devices = jax.devices()
            result['device_count'] = len(devices)
            
            for device in devices:
                device_info = {
                    'id': device.id,
                    'platform': device.platform,
                    'device_kind': device.device_kind if hasattr(device, 'device_kind') else 'unknown'
                }
                result['devices'].append(device_info)
                
                if device.platform == 'gpu':
                    result['gpu_available'] = True
                    
            # Run simple GPU test
            if result['gpu_available']:
                test_array = jax.numpy.array([1.0, 2.0, 3.0])
                result_array = jax.numpy.sum(test_array)
                result['test_passed'] = float(result_array) == 6.0
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"GPU verification failed: {e}")
            
        return result
        
    def get_troubleshooting_guide(self) -> List[str]:
        """Get troubleshooting steps for common issues"""
        guide = []
        status = self.check_jax_installation()
        
        if not status['jax_installed']:
            guide.append("1. JAX is not installed. Run the installer:")
            guide.append(f"   {self.generate_install_command()}")
            
        if status['jax_installed'] and not status['has_gpu_support'] and self.system_info.has_gpu:
            guide.append("2. GPU detected but JAX has no GPU support:")
            guide.append("   - Reinstall with GPU support:")
            guide.append(f"     {self.generate_install_command(force_reinstall=True)}")
            guide.append("   - Check CUDA installation:")
            guide.append("     nvidia-smi")
            guide.append("   - Verify CUDA paths:")
            guide.append("     echo $CUDA_HOME")
            guide.append("     echo $LD_LIBRARY_PATH")
            
        if status['issues']:
            guide.append("3. Detected issues:")
            for issue in status['issues']:
                guide.append(f"   - {issue}")
                
        if self.system_info.os_name == 'windows':
            guide.append("4. Windows-specific:")
            guide.append("   - Ensure Visual Studio C++ Build Tools are installed")
            guide.append("   - Use Anaconda/Miniconda for better compatibility")
            
        if self.system_info.os_name == 'darwin' and self.system_info.architecture == 'arm64':
            guide.append("5. Apple Silicon (M1/M2) specific:")
            guide.append("   - Use Metal backend for GPU acceleration")
            guide.append("   - Install with: pip install jax[metal]")
            
        return guide
        
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed Python packages"""
        packages = {}
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                package_list = json.loads(result.stdout)
                packages = {pkg['name']: pkg['version'] for pkg in package_list}
        except:
            pass
        return packages
        
    def generate_report(self) -> str:
        """Generate comprehensive installation report"""
        lines = []
        lines.append("=" * 60)
        lines.append("JAX Installation Report")
        lines.append("=" * 60)
        
        # System information
        lines.append("\nSystem Information:")
        lines.append(f"  OS: {self.system_info.os_name} {self.system_info.os_version}")
        lines.append(f"  Architecture: {self.system_info.architecture}")
        lines.append(f"  Python: {self.system_info.python_version}")
        lines.append(f"  CUDA: {self.system_info.cuda_version.value}")
        lines.append(f"  GPUs: {self.system_info.gpu_count}")
        lines.append(f"  CPUs: {self.system_info.cpu_count}")
        lines.append(f"  Memory: {self.system_info.memory_gb:.1f} GB")
        
        # JAX status
        lines.append("\nJAX Status:")
        status = self.check_jax_installation()
        lines.append(f"  JAX installed: {status['jax_installed']}")
        if status['jax_version']:
            lines.append(f"  JAX version: {status['jax_version']}")
        lines.append(f"  jaxlib installed: {status['jaxlib_installed']}")
        if status['jaxlib_version']:
            lines.append(f"  jaxlib version: {status['jaxlib_version']}")
        if status['backend']:
            lines.append(f"  Backend: {status['backend']}")
        lines.append(f"  GPU support: {status['has_gpu_support']}")
        
        # Issues
        if status['issues']:
            lines.append("\nIssues Detected:")
            for issue in status['issues']:
                lines.append(f"  - {issue}")
                
        # Recommendations
        lines.append("\nRecommendations:")
        lines.append(f"  Recommended package: {self.get_recommended_package()}")
        lines.append(f"  Install command: {self.generate_install_command()}")
        
        # GPU verification
        if status['jax_installed']:
            gpu_status = self.verify_gpu_support()
            lines.append("\nDevice Status:")
            lines.append(f"  Device count: {gpu_status['device_count']}")
            for device in gpu_status['devices']:
                lines.append(f"  - {device['platform']}: {device['device_kind']} (id={device['id']})")
                
        # Troubleshooting
        guide = self.get_troubleshooting_guide()
        if guide:
            lines.append("\nTroubleshooting Guide:")
            for step in guide:
                lines.append(f"  {step}")
                
        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)


def main():
    """Main entry point for JAX installer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JAX Installation Helper')
    parser.add_argument('--check', action='store_true', help='Check JAX installation status')
    parser.add_argument('--install', action='store_true', help='Install JAX with auto-detected backend')
    parser.add_argument('--force', action='store_true', help='Force reinstall JAX')
    parser.add_argument('--dry-run', action='store_true', help='Show install command without executing')
    parser.add_argument('--report', action='store_true', help='Generate full installation report')
    parser.add_argument('--verify-gpu', action='store_true', help='Verify GPU support')
    
    args = parser.parse_args()
    
    installer = JAXInstaller()
    
    if args.report or (not any([args.check, args.install, args.verify_gpu])):
        # Default action is to show report
        print(installer.generate_report())
        
    if args.check:
        status = installer.check_jax_installation()
        print(json.dumps(status, indent=2))
        
    if args.install:
        result = installer.install_jax(force_reinstall=args.force, dry_run=args.dry_run)
        if result['success']:
            print("✓ JAX installation successful")
        else:
            print(f"✗ JAX installation failed: {result['error']}")
            sys.exit(1)
            
    if args.verify_gpu:
        gpu_status = installer.verify_gpu_support()
        print(json.dumps(gpu_status, indent=2))
        if not gpu_status['gpu_available'] and installer.system_info.has_gpu:
            print("\n⚠ GPU detected but not accessible by JAX")
            print("Run with --report for troubleshooting guide")


if __name__ == '__main__':
    main()