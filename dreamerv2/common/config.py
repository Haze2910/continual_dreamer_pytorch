import json
import pathlib
import re
import sys


# Code taken 100% from original continual dreamer repo

class Config(dict):
    SEP = '.'
    IS_PATTERN = re.compile(r'.*[^A-Za-z0-9_.-].*')

    def __init__(self, *args, **kwargs):
        mapping = dict(*args, **kwargs)
        mapping = self._flatten(mapping)
        mapping = self._ensure_keys(mapping)
        mapping = self._ensure_values(mapping)
        self._flat = mapping
        self._nested = self._nest(mapping)
        # Need to assign the values to the base class dictionary so that
        # conversion to dict does not lose the content.
        super().__init__(self._nested)

    @property
    def flat(self):
        return self._flat.copy()

    def save(self, filename):
        filename = pathlib.Path(filename)
        if filename.suffix == '.json':
            filename.write_text(json.dumps(dict(self)))
        elif filename.suffix in ('.yml', '.yaml'):
            from ruamel.yaml import YAML
            with filename.open('w') as f:
                yaml = YAML(typ="safe", pure=True)
                yaml.dump(dict(self), f)
        else:
            raise NotImplementedError(filename.suffix)

    @classmethod
    def load(cls, filename):
        filename = pathlib.Path(filename)
        if filename.suffix == '.json':
            return cls(json.loads(filename.read_text()))
        elif filename.suffix in ('.yml', '.yaml'):
            import ruamel.yaml as yaml
            return cls(yaml.safe_load(filename.read_text()))
        else:
            raise NotImplementedError(filename.suffix)

    def parse_flags(self, argv=None, known_only=False, help_exists=None):
        return Flags(self).parse(argv, known_only, help_exists)

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name):
        result = self._nested
        for part in name.split(self.SEP):
            result = result[part]
        if isinstance(result, dict):
            result = type(self)(result)
        return result

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)
        message = f"Tried to set key '{key}' on immutable config. Use update()."
        raise AttributeError(message)

    def __setitem__(self, key, value):
        if key.startswith('_'):
            return super().__setitem__(key, value)
        message = f"Tried to set key '{key}' on immutable config. Use update()."
        raise AttributeError(message)

    def __reduce__(self):
        return (type(self), (dict(self),))

    def __str__(self):
        lines = ['\nConfig:']
        keys, vals, typs = [], [], []
        for key, val in self.flat.items():
            keys.append(key + ':')
            vals.append(self._format_value(val))
            typs.append(self._format_type(val))
        max_key = max(len(k) for k in keys) if keys else 0
        max_val = max(len(v) for v in vals) if vals else 0
        for key, val, typ in zip(keys, vals, typs):
            key = key.ljust(max_key)
            val = val.ljust(max_val)
            lines.append(f'{key}  {val}  ({typ})')
        return '\n'.join(lines)

    def update(self, *args, **kwargs):
        result = self._flat.copy()
        inputs = self._flatten(dict(*args, **kwargs))
        for key, new in inputs.items():
            if self.IS_PATTERN.match(key):
                pattern = re.compile(key)
                keys = {k for k in result if pattern.match(k)}
            else:
                keys = [key]
            if not keys:
                raise KeyError(f'Unknown key or pattern {key}.')
            for key in keys:
                old = result[key]
                try:
                    if isinstance(old, int) and isinstance(new, float):
                        if float(int(new)) != new:
                            message = f"Cannot convert fractional float {new} to int."
                            raise ValueError(message)
                    result[key] = type(old)(new)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"Cannot convert '{new}' to type '{type(old).__name__}' " +
                        f"of value '{old}' for key '{key}'.")
        return type(self)(result)

    def _flatten(self, mapping):
        result = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                for k, v in self._flatten(value).items():
                    if self.IS_PATTERN.match(key) or self.IS_PATTERN.match(k):
                        combined = f'{key}\\{self.SEP}{k}'
                    else:
                        combined = f'{key}{self.SEP}{k}'
                    result[combined] = v
            else:
                result[key] = value
        return result

    def _nest(self, mapping):
        result = {}
        for key, value in mapping.items():
            parts = key.split(self.SEP)
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _ensure_keys(self, mapping):
        for key in mapping:
            assert not self.IS_PATTERN.match(key), key
        return mapping

    def _ensure_values(self, mapping):
        result = json.loads(json.dumps(mapping))
        for key, value in result.items():
            if isinstance(value, list):
                value = tuple(value)
            if isinstance(value, tuple):
                if len(value) == 0:
                    message = 'Empty lists are disallowed because their type is unclear.'
                    raise TypeError(message)
                if not isinstance(value[0], (str, float, int, bool)):
                    message = 'Lists can only contain strings, floats, ints, bools'
                    message += f' but not {type(value[0])}'
                    raise TypeError(message)
                if not all(isinstance(x, type(value[0])) for x in value[1:]):
                    message = 'Elements of a list must all be of the same type.'
                    raise TypeError(message)
            result[key] = value
        return result

    def _format_value(self, value):
        if isinstance(value, (list, tuple)):
            return '[' + ', '.join(self._format_value(x) for x in value) + ']'
        return str(value)

    def _format_type(self, value):
        if isinstance(value, (list, tuple)):
            assert len(value) > 0, value
            return self._format_type(value[0]) + 's'
        return str(type(value).__name__)

class Flags:

    def __init__(self, *args, **kwargs):
        from .config import Config
        self._config = Config(*args, **kwargs)

    def parse(self, argv=None, known_only=False, help_exists=None):
        if help_exists is None:
            help_exists = not known_only
        if argv is None:
            # this causes an error as all sys args like lists and bools
            # are not parsed correctly here. Let's manually update the config
            # dictionary with args in train_{env}.py
            # argv = sys.argv[1:]
            argv = []
        if '--help' in argv:
            print('\nHelp:')
            lines = str(self._config).split('\n')[2:]
            print('\n'.join('--' + re.sub(r'[:,\[\]]', '', x) for x in lines))
            help_exists and sys.exit()
        parsed = {}
        remaining = []
        key = None
        vals = None
        for arg in argv:
            if arg.startswith('--'):
                if key:
                    self._submit_entry(key, vals, parsed, remaining)
                if '=' in arg:
                    key, val = arg.split('=', 1)
                    vals = [val]
                else:
                    key, vals = arg, []
            else:
                if key:
                    vals.append(arg)
                else:
                    remaining.append(arg)
        self._submit_entry(key, vals, parsed, remaining)
        parsed = self._config.update(parsed)
        if known_only:
            return parsed, remaining
        else:
            for flag in remaining:
                if flag.startswith('--'):
                    raise ValueError(f"Flag '{flag}' did not match any config keys.")
            assert not remaining, remaining
            return parsed

    def _submit_entry(self, key, vals, parsed, remaining):
        if not key and not vals:
            return
        if not key:
            vals = ', '.join(f"'{x}'" for x in vals)
            raise ValueError(f"Values {vals} were not preceeded by any flag.")
        name = key[len('--'):]
        if '=' in name:
            remaining.extend([key] + vals)
            return
        if self._config.IS_PATTERN.match(name):
            pattern = re.compile(name)
            keys = {k for k in self._config.flat if pattern.match(k)}
        elif name in self._config:
            keys = [name]
        else:
            keys = []
        if not keys:
            remaining.extend([key] + vals)
            return
        if not vals:
            raise ValueError(f"Flag '{key}' was not followed by any values.")
        for key in keys:
            parsed[key] = self._parse_flag_value(self._config[key], vals, key)

    def _parse_flag_value(self, default, value, key):
        value = value if isinstance(value, (tuple, list)) else (value,)
        if isinstance(default, (tuple, list)):
            if len(value) == 1 and ',' in value[0]:
                value = value[0].split(',')
            return tuple(self._parse_flag_value(default[0], [x], key) for x in value)
        assert len(value) == 1, value
        value = str(value[0])
        if default is None:
            return value
        if isinstance(default, bool):
            try:
                return bool(['False', 'True'].index(value))
            except ValueError:
                message = f"Expected bool but got '{value}' for key '{key}'."
                raise TypeError(message)
        if isinstance(default, int):
            value = float(value)  # Allow scientific notation for integers.
            if float(int(value)) != value:
                message = f"Expected int but got float '{value}' for key '{key}'."
                raise TypeError(message)
            return int(value)
        return type(default)(value)