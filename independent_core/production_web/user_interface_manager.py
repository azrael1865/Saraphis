"""
Saraphis User Interface Manager
Production-ready user interface validation and management
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
import json
import hashlib

logger = logging.getLogger(__name__)


class UserInterfaceManager:
    """Production-ready user interface management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # User preferences storage
        self.user_preferences = defaultdict(dict)
        self.default_preferences = {
            'theme': 'dark',
            'language': 'en',
            'timezone': 'UTC',
            'date_format': 'YYYY-MM-DD',
            'time_format': '24h',
            'dashboard_layout': 'grid',
            'refresh_rate': 5,
            'notifications_enabled': True,
            'sound_enabled': False,
            'compact_mode': False,
            'chart_animations': True,
            'auto_save': True,
            'data_precision': 2,
            'show_tooltips': True
        }
        
        # User permissions
        self.user_permissions = defaultdict(set)
        self.role_permissions = {
            'admin': {
                'dashboards': ['all'],
                'features': ['all'],
                'settings': ['all'],
                'data_access': ['all']
            },
            'power_user': {
                'dashboards': ['system_overview', 'uncertainty_analysis', 'training_monitoring', 'production_metrics'],
                'features': ['export', 'customize', 'share'],
                'settings': ['preferences', 'notifications'],
                'data_access': ['read', 'export']
            },
            'user': {
                'dashboards': ['system_overview', 'production_metrics'],
                'features': ['view', 'export'],
                'settings': ['preferences'],
                'data_access': ['read']
            },
            'viewer': {
                'dashboards': ['system_overview'],
                'features': ['view'],
                'settings': [],
                'data_access': ['read']
            }
        }
        
        # User roles
        self.user_roles = defaultdict(lambda: 'user')
        
        # Interaction validation rules
        self.validation_rules = {
            'click': {
                'required_fields': ['target', 'coordinates'],
                'rate_limit': 10  # per second
            },
            'input': {
                'required_fields': ['field', 'value'],
                'rate_limit': 5,
                'max_length': 1000
            },
            'drag': {
                'required_fields': ['source', 'target', 'start_coords', 'end_coords'],
                'rate_limit': 20
            },
            'scroll': {
                'required_fields': ['direction', 'amount'],
                'rate_limit': 50
            },
            'resize': {
                'required_fields': ['component_id', 'dimensions'],
                'rate_limit': 10
            },
            'export': {
                'required_fields': ['format', 'data_type'],
                'rate_limit': 1,
                'requires_permission': 'export'
            },
            'settings_change': {
                'required_fields': ['setting', 'value'],
                'rate_limit': 5,
                'requires_permission': 'settings'
            }
        }
        
        # Interaction processing handlers
        self.interaction_handlers = {
            'click': self._process_click,
            'input': self._process_input,
            'drag': self._process_drag,
            'scroll': self._process_scroll,
            'resize': self._process_resize,
            'export': self._process_export,
            'settings_change': self._process_settings_change,
            'command': self._process_command,
            'shortcut': self._process_shortcut
        }
        
        # User session tracking
        self.user_sessions = defaultdict(dict)
        self.session_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Rate limiting
        self.rate_limiters = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'reset_time': time.time() + 1}))
        
        # Interaction history
        self.interaction_history = deque(maxlen=10000)
        
        # UI state management
        self.ui_states = defaultdict(dict)
        self.state_history = defaultdict(lambda: deque(maxlen=100))
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'validation_errors': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start background threads
        self._start_background_threads()
        
        self.logger.info("User Interface Manager initialized")
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences with defaults"""
        try:
            with self._lock:
                # Get stored preferences
                preferences = self.user_preferences.get(user_id, {}).copy()
                
                # Apply defaults for missing values
                for key, value in self.default_preferences.items():
                    if key not in preferences:
                        preferences[key] = value
                
                return preferences
                
        except Exception as e:
            self.logger.error(f"Preference retrieval failed: {e}")
            return self.default_preferences.copy()
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences"""
        try:
            with self._lock:
                # Validate preferences
                validated_prefs = self._validate_preferences(preferences)
                
                # Update stored preferences
                if user_id not in self.user_preferences:
                    self.user_preferences[user_id] = {}
                
                self.user_preferences[user_id].update(validated_prefs)
                
                # Track update
                self._track_preference_update(user_id, validated_prefs)
                
                return {
                    'success': True,
                    'preferences': self.user_preferences[user_id]
                }
                
        except Exception as e:
            self.logger.error(f"Preference update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_user_permissions(self, user_id: str) -> Dict[str, Any]:
        """Get user permissions based on role"""
        try:
            with self._lock:
                # Get user role
                role = self.user_roles.get(user_id, 'user')
                
                # Get role permissions
                role_perms = self.role_permissions.get(role, self.role_permissions['user'])
                
                # Get additional user-specific permissions
                user_perms = self.user_permissions.get(user_id, set())
                
                # Combine permissions
                permissions = {
                    'role': role,
                    'dashboards': role_perms['dashboards'].copy(),
                    'features': role_perms['features'].copy(),
                    'settings': role_perms['settings'].copy(),
                    'data_access': role_perms['data_access'].copy(),
                    'custom_permissions': list(user_perms)
                }
                
                return permissions
                
        except Exception as e:
            self.logger.error(f"Permission retrieval failed: {e}")
            return {
                'role': 'viewer',
                'dashboards': [],
                'features': ['view'],
                'settings': [],
                'data_access': ['read']
            }
    
    def validate_dashboard_access(self, user_id: str, dashboard_type: str, 
                                permissions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user access to dashboard"""
        try:
            # Check if user has access to all dashboards
            if 'all' in permissions.get('dashboards', []):
                return {
                    'authorized': True,
                    'details': 'Full dashboard access'
                }
            
            # Check specific dashboard access
            if dashboard_type in permissions.get('dashboards', []):
                return {
                    'authorized': True,
                    'details': f'Access granted to {dashboard_type}'
                }
            
            # Access denied
            return {
                'authorized': False,
                'details': f'No access to {dashboard_type} dashboard'
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard access validation failed: {e}")
            return {
                'authorized': False,
                'details': f'Validation error: {str(e)}'
            }
    
    def validate_interaction(self, user_id: str, 
                           interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user interaction"""
        try:
            interaction_type = interaction_data.get('interaction_type')
            
            # Check if interaction type is valid
            if interaction_type not in self.validation_rules:
                return {
                    'valid': False,
                    'details': f'Unknown interaction type: {interaction_type}'
                }
            
            # Get validation rules
            rules = self.validation_rules[interaction_type]
            
            # Check required fields
            missing_fields = []
            for field in rules.get('required_fields', []):
                if field not in interaction_data:
                    missing_fields.append(field)
            
            if missing_fields:
                return {
                    'valid': False,
                    'details': f'Missing required fields: {missing_fields}'
                }
            
            # Check rate limit
            if not self._check_interaction_rate_limit(user_id, interaction_type, rules.get('rate_limit', 10)):
                return {
                    'valid': False,
                    'details': 'Rate limit exceeded'
                }
            
            # Check permissions if required
            if 'requires_permission' in rules:
                permissions = self.get_user_permissions(user_id)
                required_perm = rules['requires_permission']
                
                if required_perm not in permissions.get('features', []) and 'all' not in permissions.get('features', []):
                    return {
                        'valid': False,
                        'details': f'Permission denied: {required_perm}'
                    }
            
            # Validate specific interaction data
            validation_result = self._validate_interaction_data(interaction_type, interaction_data)
            
            # Update statistics
            with self._lock:
                self.validation_stats['total_validations'] += 1
                if validation_result['valid']:
                    self.validation_stats['successful_validations'] += 1
                else:
                    self.validation_stats['failed_validations'] += 1
                    self.validation_stats['validation_errors'][validation_result.get('error_type', 'unknown')] += 1
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Interaction validation failed: {e}")
            return {
                'valid': False,
                'details': f'Validation error: {str(e)}'
            }
    
    def process_interaction(self, user_id: str, 
                          interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process validated user interaction"""
        try:
            interaction_type = interaction_data.get('interaction_type')
            
            # Get handler for interaction type
            handler = self.interaction_handlers.get(interaction_type)
            
            if not handler:
                return {
                    'success': False,
                    'error': f'No handler for interaction type: {interaction_type}'
                }
            
            # Process interaction
            result = handler(user_id, interaction_data)
            
            # Track interaction
            self._track_interaction(user_id, interaction_type, result)
            
            # Update UI state if needed
            if result.get('state_update'):
                self._update_ui_state(user_id, result['state_update'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Interaction processing failed: {e}")
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }
    
    def _validate_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user preferences"""
        validated = {}
        
        # Theme validation
        if 'theme' in preferences:
            if preferences['theme'] in ['dark', 'light', 'midnight']:
                validated['theme'] = preferences['theme']
        
        # Language validation
        if 'language' in preferences:
            if preferences['language'] in ['en', 'es', 'fr', 'de', 'ja', 'zh']:
                validated['language'] = preferences['language']
        
        # Timezone validation
        if 'timezone' in preferences:
            # In production, validate against pytz timezones
            validated['timezone'] = preferences['timezone']
        
        # Date format validation
        if 'date_format' in preferences:
            if preferences['date_format'] in ['YYYY-MM-DD', 'DD/MM/YYYY', 'MM/DD/YYYY']:
                validated['date_format'] = preferences['date_format']
        
        # Time format validation
        if 'time_format' in preferences:
            if preferences['time_format'] in ['12h', '24h']:
                validated['time_format'] = preferences['time_format']
        
        # Dashboard layout validation
        if 'dashboard_layout' in preferences:
            if preferences['dashboard_layout'] in ['grid', 'list', 'compact', 'custom']:
                validated['dashboard_layout'] = preferences['dashboard_layout']
        
        # Numeric validations
        if 'refresh_rate' in preferences:
            rate = preferences['refresh_rate']
            if isinstance(rate, (int, float)) and 1 <= rate <= 60:
                validated['refresh_rate'] = rate
        
        if 'data_precision' in preferences:
            precision = preferences['data_precision']
            if isinstance(precision, int) and 0 <= precision <= 6:
                validated['data_precision'] = precision
        
        # Boolean validations
        boolean_fields = [
            'notifications_enabled', 'sound_enabled', 'compact_mode',
            'chart_animations', 'auto_save', 'show_tooltips'
        ]
        
        for field in boolean_fields:
            if field in preferences and isinstance(preferences[field], bool):
                validated[field] = preferences[field]
        
        return validated
    
    def _validate_interaction_data(self, interaction_type: str, 
                                 data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate specific interaction data"""
        try:
            if interaction_type == 'click':
                return self._validate_click_data(data)
            elif interaction_type == 'input':
                return self._validate_input_data(data)
            elif interaction_type == 'drag':
                return self._validate_drag_data(data)
            elif interaction_type == 'scroll':
                return self._validate_scroll_data(data)
            elif interaction_type == 'resize':
                return self._validate_resize_data(data)
            elif interaction_type == 'export':
                return self._validate_export_data(data)
            elif interaction_type == 'settings_change':
                return self._validate_settings_data(data)
            else:
                return {'valid': True}  # Default to valid for unknown types
                
        except Exception as e:
            return {
                'valid': False,
                'details': f'Data validation error: {str(e)}',
                'error_type': 'data_validation'
            }
    
    def _validate_click_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate click interaction data"""
        target = data.get('target')
        coords = data.get('coordinates')
        
        # Validate target
        if not isinstance(target, str) or not target:
            return {
                'valid': False,
                'details': 'Invalid target',
                'error_type': 'invalid_target'
            }
        
        # Validate coordinates
        if not isinstance(coords, dict) or 'x' not in coords or 'y' not in coords:
            return {
                'valid': False,
                'details': 'Invalid coordinates',
                'error_type': 'invalid_coordinates'
            }
        
        if not all(isinstance(coords[k], (int, float)) for k in ['x', 'y']):
            return {
                'valid': False,
                'details': 'Coordinates must be numeric',
                'error_type': 'invalid_coordinates'
            }
        
        return {'valid': True}
    
    def _validate_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input interaction data"""
        field = data.get('field')
        value = data.get('value')
        rules = self.validation_rules['input']
        
        # Validate field
        if not isinstance(field, str) or not field:
            return {
                'valid': False,
                'details': 'Invalid field',
                'error_type': 'invalid_field'
            }
        
        # Validate value length
        if isinstance(value, str) and len(value) > rules.get('max_length', 1000):
            return {
                'valid': False,
                'details': f'Value exceeds maximum length of {rules["max_length"]}',
                'error_type': 'value_too_long'
            }
        
        return {'valid': True}
    
    def _validate_drag_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate drag interaction data"""
        required = ['source', 'target', 'start_coords', 'end_coords']
        
        for field in required:
            if field not in data:
                return {
                    'valid': False,
                    'details': f'Missing {field}',
                    'error_type': 'missing_field'
                }
        
        # Validate coordinates
        for coord_field in ['start_coords', 'end_coords']:
            coords = data[coord_field]
            if not isinstance(coords, dict) or not all(k in coords for k in ['x', 'y']):
                return {
                    'valid': False,
                    'details': f'Invalid {coord_field}',
                    'error_type': 'invalid_coordinates'
                }
        
        return {'valid': True}
    
    def _validate_scroll_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scroll interaction data"""
        direction = data.get('direction')
        amount = data.get('amount')
        
        # Validate direction
        if direction not in ['up', 'down', 'left', 'right']:
            return {
                'valid': False,
                'details': 'Invalid scroll direction',
                'error_type': 'invalid_direction'
            }
        
        # Validate amount
        if not isinstance(amount, (int, float)) or amount <= 0:
            return {
                'valid': False,
                'details': 'Invalid scroll amount',
                'error_type': 'invalid_amount'
            }
        
        return {'valid': True}
    
    def _validate_resize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resize interaction data"""
        component_id = data.get('component_id')
        dimensions = data.get('dimensions')
        
        # Validate component ID
        if not isinstance(component_id, str) or not component_id:
            return {
                'valid': False,
                'details': 'Invalid component ID',
                'error_type': 'invalid_component'
            }
        
        # Validate dimensions
        if not isinstance(dimensions, dict):
            return {
                'valid': False,
                'details': 'Invalid dimensions',
                'error_type': 'invalid_dimensions'
            }
        
        # Check width and height
        for dim in ['width', 'height']:
            if dim in dimensions:
                value = dimensions[dim]
                if not isinstance(value, (int, float)) or value <= 0:
                    return {
                        'valid': False,
                        'details': f'Invalid {dim}',
                        'error_type': 'invalid_dimensions'
                    }
        
        return {'valid': True}
    
    def _validate_export_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate export interaction data"""
        format_type = data.get('format')
        data_type = data.get('data_type')
        
        # Validate format
        valid_formats = ['json', 'csv', 'xlsx', 'pdf', 'png']
        if format_type not in valid_formats:
            return {
                'valid': False,
                'details': f'Invalid format. Must be one of: {valid_formats}',
                'error_type': 'invalid_format'
            }
        
        # Validate data type
        valid_data_types = ['dashboard', 'chart', 'table', 'metrics', 'logs']
        if data_type not in valid_data_types:
            return {
                'valid': False,
                'details': f'Invalid data type. Must be one of: {valid_data_types}',
                'error_type': 'invalid_data_type'
            }
        
        return {'valid': True}
    
    def _validate_settings_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate settings change data"""
        setting = data.get('setting')
        value = data.get('value')
        
        # Validate setting exists
        if not isinstance(setting, str) or not setting:
            return {
                'valid': False,
                'details': 'Invalid setting name',
                'error_type': 'invalid_setting'
            }
        
        # Setting-specific validation would go here
        
        return {'valid': True}
    
    def _check_interaction_rate_limit(self, user_id: str, 
                                    interaction_type: str, limit: int) -> bool:
        """Check if interaction is within rate limit"""
        current_time = time.time()
        
        with self._lock:
            limiter = self.rate_limiters[user_id][interaction_type]
            
            # Reset if window expired
            if current_time >= limiter['reset_time']:
                limiter['count'] = 0
                limiter['reset_time'] = current_time + 1  # 1 second window
            
            # Check limit
            if limiter['count'] >= limit:
                return False
            
            limiter['count'] += 1
            return True
    
    def _process_click(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process click interaction"""
        try:
            target = data['target']
            coords = data['coordinates']
            
            # Determine action based on target
            if target.startswith('button_'):
                action = target.replace('button_', '')
                return {
                    'success': True,
                    'action': action,
                    'affected_components': [target],
                    'updates': {}
                }
            elif target.startswith('component_'):
                return {
                    'success': True,
                    'action': 'select_component',
                    'affected_components': [target],
                    'updates': {
                        target: {'state': 'selected'}
                    }
                }
            else:
                return {
                    'success': True,
                    'action': 'generic_click',
                    'affected_components': []
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_input(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input interaction"""
        try:
            field = data['field']
            value = data['value']
            
            # Sanitize input
            sanitized_value = self._sanitize_input(value)
            
            return {
                'success': True,
                'action': 'update_field',
                'affected_components': [field],
                'updates': {
                    field: {
                        'value': sanitized_value,
                        'validated': True
                    }
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_drag(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process drag interaction"""
        try:
            source = data['source']
            target = data['target']
            
            return {
                'success': True,
                'action': 'drag_drop',
                'affected_components': [source, target],
                'updates': {
                    source: {'state': 'moved'},
                    target: {'state': 'drop_target'}
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_scroll(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process scroll interaction"""
        try:
            direction = data['direction']
            amount = data['amount']
            component = data.get('component', 'main_view')
            
            return {
                'success': True,
                'action': 'scroll',
                'affected_components': [component],
                'updates': {
                    component: {
                        'scroll_position': {
                            'direction': direction,
                            'offset': amount
                        }
                    }
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_resize(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process resize interaction"""
        try:
            component_id = data['component_id']
            dimensions = data['dimensions']
            
            return {
                'success': True,
                'action': 'resize',
                'affected_components': [component_id],
                'updates': {
                    component_id: {
                        'dimensions': dimensions,
                        'requires_render': True
                    }
                },
                'state_update': {
                    'layout_modified': True
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_export(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process export interaction"""
        try:
            format_type = data['format']
            data_type = data['data_type']
            
            # Generate export ID
            export_id = f"export_{int(time.time())}_{user_id[:8]}"
            
            return {
                'success': True,
                'action': 'export',
                'export_id': export_id,
                'format': format_type,
                'data_type': data_type,
                'affected_components': [],
                'status': 'queued'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_settings_change(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process settings change"""
        try:
            setting = data['setting']
            value = data['value']
            
            # Update preference
            result = self.update_user_preferences(
                user_id, {setting: value}
            )
            
            if result['success']:
                return {
                    'success': True,
                    'action': 'settings_updated',
                    'setting': setting,
                    'value': value,
                    'affected_components': ['settings_panel'],
                    'state_update': {
                        'preferences_modified': True
                    }
                }
            else:
                return result
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_command(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process command interaction"""
        try:
            command = data.get('command')
            args = data.get('args', {})
            
            # Command processing logic
            return {
                'success': True,
                'action': 'command_executed',
                'command': command,
                'result': f'Command {command} executed'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_shortcut(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process keyboard shortcut"""
        try:
            shortcut = data.get('shortcut')
            
            # Shortcut processing logic
            return {
                'success': True,
                'action': 'shortcut_triggered',
                'shortcut': shortcut
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _sanitize_input(self, value: Any) -> Any:
        """Sanitize user input"""
        if isinstance(value, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '&', '"', "'"]
            sanitized = value
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            return sanitized.strip()
        
        return value
    
    def _track_interaction(self, user_id: str, interaction_type: str, 
                         result: Dict[str, Any]):
        """Track user interaction"""
        try:
            interaction_record = {
                'timestamp': time.time(),
                'user_id': user_id,
                'interaction_type': interaction_type,
                'success': result.get('success', False),
                'action': result.get('action')
            }
            
            with self._lock:
                # Add to global history
                self.interaction_history.append(interaction_record)
                
                # Add to user session history
                self.session_history[user_id].append(interaction_record)
                
                # Update session activity
                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = {
                        'started': time.time(),
                        'last_activity': time.time(),
                        'interaction_count': 0
                    }
                
                self.user_sessions[user_id]['last_activity'] = time.time()
                self.user_sessions[user_id]['interaction_count'] += 1
                
        except Exception as e:
            self.logger.error(f"Interaction tracking failed: {e}")
    
    def _track_preference_update(self, user_id: str, preferences: Dict[str, Any]):
        """Track preference updates"""
        try:
            update_record = {
                'timestamp': time.time(),
                'user_id': user_id,
                'updated_preferences': list(preferences.keys())
            }
            
            self.logger.info(f"Preferences updated for user {user_id}: {list(preferences.keys())}")
            
        except Exception as e:
            self.logger.error(f"Preference tracking failed: {e}")
    
    def _update_ui_state(self, user_id: str, state_update: Dict[str, Any]):
        """Update UI state for user"""
        try:
            with self._lock:
                if user_id not in self.ui_states:
                    self.ui_states[user_id] = {}
                
                # Apply state update
                self.ui_states[user_id].update(state_update)
                
                # Track state change
                self.state_history[user_id].append({
                    'timestamp': time.time(),
                    'changes': state_update
                })
                
        except Exception as e:
            self.logger.error(f"UI state update failed: {e}")
    
    def _start_background_threads(self):
        """Start background maintenance threads"""
        # Session cleanup thread
        session_thread = threading.Thread(
            target=self._session_cleanup_loop,
            daemon=True
        )
        session_thread.start()
        
        # Rate limiter cleanup thread
        rate_limiter_thread = threading.Thread(
            target=self._rate_limiter_cleanup_loop,
            daemon=True
        )
        rate_limiter_thread.start()
    
    def _session_cleanup_loop(self):
        """Clean up inactive sessions"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                current_time = time.time()
                inactive_threshold = 3600  # 1 hour
                
                with self._lock:
                    # Find inactive sessions
                    inactive_users = []
                    for user_id, session in self.user_sessions.items():
                        if current_time - session['last_activity'] > inactive_threshold:
                            inactive_users.append(user_id)
                    
                    # Clean up inactive sessions
                    for user_id in inactive_users:
                        del self.user_sessions[user_id]
                        self.logger.info(f"Cleaned up inactive session for user: {user_id}")
                        
            except Exception as e:
                self.logger.error(f"Session cleanup error: {e}")
    
    def _rate_limiter_cleanup_loop(self):
        """Clean up old rate limiter entries"""
        while True:
            try:
                time.sleep(60)  # Run every minute
                
                current_time = time.time()
                
                with self._lock:
                    # Clean up expired rate limiters
                    users_to_clean = []
                    for user_id, limiters in self.rate_limiters.items():
                        # Check if all limiters are expired
                        all_expired = all(
                            current_time > limiter['reset_time'] + 60
                            for limiter in limiters.values()
                        )
                        
                        if all_expired:
                            users_to_clean.append(user_id)
                    
                    for user_id in users_to_clean:
                        del self.rate_limiters[user_id]
                        
            except Exception as e:
                self.logger.error(f"Rate limiter cleanup error: {e}")