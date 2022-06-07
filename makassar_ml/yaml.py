import os
import re
import yaml

# Extract environment variables of the form `$ENV` or `${ENV}`.
path_matcher = re.compile(r'.*(\$\{([^}^{]+)\})|(\$([^}^{]+)).*')
def path_constructor(loader, node):
    return os.path.expandvars(node.value)

# Concatenate strings.
def join_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(item) for item in seq])

# Configuration loader with custom tags.
class ConfigLoader(yaml.SafeLoader):
    pass
ConfigLoader.add_implicit_resolver('!path', path_matcher, None)
ConfigLoader.add_constructor('!path', path_constructor)
ConfigLoader.add_constructor('!join', join_constructor)