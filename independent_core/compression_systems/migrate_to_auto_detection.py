#!/usr/bin/env python3
"""
Migration Script - Update all hardcoded GPU references to use auto-detection
This script updates all files in the compression system to use the new GPU auto-detection system
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import time


class GPUReferenceMigrator:
    """Migrate hardcoded GPU references to auto-detection"""
    
    def __init__(self, dry_run: bool = True, verbose: bool = False):
        """Initialize migrator
        
        Args:
            dry_run: If True, only show what would be changed without modifying files
            verbose: If True, show detailed output
        """
        self.dry_run = dry_run
        self.verbose = verbose
        self.changes_made = []
        self.files_modified = set()
        
        # Pattern mappings for replacements
        self.patterns = {
            # RTX 5060 Ti references
            r'RTX 5060 Ti': 'detected GPU',
            r'5060 Ti': 'detected GPU',
            
            # Hardcoded memory values
            r'gpu_memory_threshold_mb:\s*int\s*=\s*2048': 
                'gpu_memory_threshold_mb: int = None  # Auto-detected',
            r'gpu_memory_limit_mb:\s*int\s*=\s*14336':
                'gpu_memory_limit_mb: int = None  # Auto-detected',
            r'gpu_critical_threshold_mb:\s*int\s*=\s*13312':
                'gpu_critical_threshold_mb: int = None  # Auto-detected',
            r'gpu_high_threshold_mb:\s*int\s*=\s*10240':
                'gpu_high_threshold_mb: int = None  # Auto-detected',
            r'gpu_moderate_threshold_mb:\s*int\s*=\s*6144':
                'gpu_moderate_threshold_mb: int = None  # Auto-detected',
            
            # Memory pressure thresholds
            r'memory_pressure_threshold:\s*float\s*=\s*0\.90':
                'memory_pressure_threshold: float = None  # Auto-detected',
            r'gpu_critical_utilization:\s*float\s*=\s*0\.95':
                'gpu_critical_utilization: float = None  # Auto-detected',
            r'gpu_high_utilization:\s*float\s*=\s*0\.85':
                'gpu_high_utilization: float = None  # Auto-detected',
            r'gpu_moderate_utilization:\s*float\s*=\s*0\.70':
                'gpu_moderate_utilization: float = None  # Auto-detected',
            
            # NUMA nodes
            r'list\(range\(16\)\)': 'None  # Auto-detected NUMA nodes',
            r'numa_nodes:\s*List\[int\]\s*=\s*field\(default_factory=lambda:\s*list\(range\(16\)\)\)':
                'numa_nodes: List[int] = None  # Auto-detected',
            
            # Batch sizes
            r'cpu_batch_size:\s*int\s*=\s*10000':
                'cpu_batch_size: int = None  # Auto-detected',
            r'cpu_batch_size:\s*int\s*=\s*50000':
                'cpu_batch_size: int = None  # Auto-detected',
            r'max_cpu_batch_size:\s*int\s*=\s*100000':
                'max_cpu_batch_size: int = None  # Auto-detected',
            r'chunk_size:\s*int\s*=\s*25000':
                'chunk_size: int = None  # Auto-detected',
            
            # Other parameters
            r'burst_multiplier:\s*float\s*=\s*5\.0':
                'burst_multiplier: float = None  # Auto-detected',
            r'emergency_cpu_workers:\s*int\s*=\s*100':
                'emergency_cpu_workers: int = None  # Auto-detected',
            r'memory_defrag_threshold:\s*float\s*=\s*0\.3':
                'memory_defrag_threshold: float = None  # Auto-detected',
        }
        
        # Files to skip
        self.skip_files = {
            'gpu_auto_detector.py',
            'test_gpu_auto_detector.py',
            'migrate_to_auto_detection.py',
            '__pycache__',
            '.pyc',
            '.md',
            '.txt',
            '.json'
        }
    
    def should_skip_file(self, filepath: Path) -> bool:
        """Check if file should be skipped
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file should be skipped
        """
        # Skip if in skip list
        for skip_pattern in self.skip_files:
            if skip_pattern in str(filepath):
                return True
        
        # Only process Python files
        if not filepath.suffix == '.py':
            return True
        
        return False
    
    def find_hardcoded_references(self, content: str) -> List[Tuple[str, str, int]]:
        """Find all hardcoded references in content
        
        Args:
            content: File content
            
        Returns:
            List of (pattern, replacement, line_number) tuples
        """
        matches = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, replacement in self.patterns.items():
                if re.search(pattern, line):
                    matches.append((pattern, replacement, line_num))
        
        return matches
    
    def update_file_content(self, content: str) -> Tuple[str, int]:
        """Update file content with auto-detection
        
        Args:
            content: Original file content
            
        Returns:
            Tuple of (updated_content, num_changes)
        """
        updated_content = content
        num_changes = 0
        
        for pattern, replacement in self.patterns.items():
            updated, count = re.subn(pattern, replacement, updated_content)
            if count > 0:
                updated_content = updated
                num_changes += count
                if self.verbose:
                    print(f"  Replaced {count} instances of pattern: {pattern}")
        
        return updated_content, num_changes
    
    def add_imports_if_needed(self, content: str, filepath: Path) -> str:
        """Add necessary imports for auto-detection
        
        Args:
            content: File content
            filepath: Path to file
            
        Returns:
            Updated content with imports
        """
        # Check if file needs auto-detection imports
        needs_import = False
        
        # Check for config classes that need auto-detection
        config_classes = [
            'CPUBurstingConfig',
            'PressureHandlerConfig',
            'SystemConfiguration'
        ]
        
        for class_name in config_classes:
            if f'class {class_name}' in content:
                needs_import = True
                break
        
        if not needs_import:
            return content
        
        # Check if import already exists
        if 'gpu_auto_detector' in content:
            return content
        
        # Determine import path based on file location
        rel_path = filepath.relative_to(Path('/Users/will/Desktop/trueSaraphis/independent_core/compression_systems'))
        depth = len(rel_path.parts) - 1
        
        if depth == 0:
            import_path = '.gpu_memory.gpu_auto_detector'
        elif depth == 1:
            if rel_path.parts[0] == 'gpu_memory':
                import_path = '.gpu_auto_detector'
            elif rel_path.parts[0] == 'padic':
                import_path = '..gpu_memory.gpu_auto_detector'
            else:
                import_path = '..gpu_memory.gpu_auto_detector'
        else:
            import_path = '../' * (depth - 1) + 'gpu_memory.gpu_auto_detector'
        
        # Add import after other imports
        import_line = f"from {import_path} import get_config_updater\n"
        
        # Find location to insert import
        lines = content.split('\n')
        import_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                import_idx = i + 1
            elif import_idx > 0 and not line.strip().startswith(('from', 'import', '#')):
                break
        
        lines.insert(import_idx, import_line)
        return '\n'.join(lines)
    
    def migrate_file(self, filepath: Path) -> bool:
        """Migrate a single file
        
        Args:
            filepath: Path to file to migrate
            
        Returns:
            True if file was modified
        """
        if self.should_skip_file(filepath):
            return False
        
        try:
            # Read file content
            with open(filepath, 'r') as f:
                original_content = f.read()
            
            # Find hardcoded references
            matches = self.find_hardcoded_references(original_content)
            
            if not matches:
                return False
            
            print(f"\n{'[DRY RUN] ' if self.dry_run else ''}Processing: {filepath}")
            
            if self.verbose:
                print(f"  Found {len(matches)} hardcoded references:")
                for pattern, replacement, line_num in matches:
                    print(f"    Line {line_num}: {pattern[:50]}...")
            
            # Update content
            updated_content, num_changes = self.update_file_content(original_content)
            
            # Add imports if needed
            updated_content = self.add_imports_if_needed(updated_content, filepath)
            
            if num_changes > 0:
                if not self.dry_run:
                    # Write updated content
                    with open(filepath, 'w') as f:
                        f.write(updated_content)
                    print(f"  ✓ Updated {num_changes} references")
                else:
                    print(f"  Would update {num_changes} references")
                
                self.files_modified.add(str(filepath))
                self.changes_made.append({
                    'file': str(filepath),
                    'changes': num_changes,
                    'matches': matches
                })
                
                return True
            
        except Exception as e:
            print(f"  ✗ Error processing {filepath}: {e}")
        
        return False
    
    def migrate_directory(self, directory: Path) -> None:
        """Migrate all files in directory
        
        Args:
            directory: Directory to migrate
        """
        print(f"{'[DRY RUN] ' if self.dry_run else ''}Migrating files in: {directory}")
        
        # Find all Python files
        python_files = list(directory.rglob('*.py'))
        print(f"Found {len(python_files)} Python files")
        
        # Process each file
        for filepath in python_files:
            self.migrate_file(filepath)
    
    def generate_report(self) -> str:
        """Generate migration report
        
        Returns:
            Report string
        """
        report = []
        report.append("\n" + "=" * 60)
        report.append("Migration Report")
        report.append("=" * 60)
        
        if self.dry_run:
            report.append("MODE: DRY RUN (no files modified)")
        else:
            report.append("MODE: LIVE (files modified)")
        
        report.append(f"\nFiles modified: {len(self.files_modified)}")
        report.append(f"Total changes: {sum(c['changes'] for c in self.changes_made)}")
        
        if self.files_modified:
            report.append("\nModified files:")
            for filepath in sorted(self.files_modified):
                report.append(f"  - {filepath}")
        
        if self.verbose and self.changes_made:
            report.append("\nDetailed changes:")
            for change in self.changes_made:
                report.append(f"\n  {change['file']}:")
                for pattern, replacement, line_num in change['matches']:
                    report.append(f"    Line {line_num}: {pattern[:50]}...")
        
        report.append("\n" + "=" * 60)
        
        return '\n'.join(report)
    
    def save_backup(self, directory: Path) -> None:
        """Create backup of all files before migration
        
        Args:
            directory: Directory to backup
        """
        backup_dir = directory / '.migration_backup'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_file = backup_dir / f'backup_{timestamp}.json'
        
        backups = {}
        for filepath in directory.rglob('*.py'):
            if not self.should_skip_file(filepath):
                with open(filepath, 'r') as f:
                    backups[str(filepath)] = f.read()
        
        with open(backup_file, 'w') as f:
            json.dump(backups, f)
        
        print(f"Backup saved to: {backup_file}")


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description='Migrate hardcoded GPU references to auto-detection'
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='/Users/will/Desktop/trueSaraphis/independent_core/compression_systems',
        help='Directory to migrate'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually modify files (disables dry-run)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before migration'
    )
    
    args = parser.parse_args()
    
    # Override dry-run if execute is specified
    if args.execute:
        args.dry_run = False
    
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}")
        sys.exit(1)
    
    # Create migrator
    migrator = GPUReferenceMigrator(
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        migrator.save_backup(directory)
    
    # Run migration
    migrator.migrate_directory(directory)
    
    # Generate and print report
    report = migrator.generate_report()
    print(report)
    
    # Summary
    if args.dry_run:
        print("\nTo execute the migration, run with --execute flag")
    else:
        print("\nMigration complete!")


if __name__ == "__main__":
    main()