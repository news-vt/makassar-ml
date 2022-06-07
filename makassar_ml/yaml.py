import os
import re
import yaml

# Extract environment variables of the form `$ENV` or `${ENV}`.
path_matcher = re.compile(r'.*(\$\{([^}^{]+)\})|(\$([^}^{]+)).*')
def path_constructor(loader, node):
    return os.path.expandvars(node.value)

# Loader for environment variables within YAML files.
class EnvVarLoader(yaml.SafeLoader):
    pass
EnvVarLoader.add_implicit_resolver('!path', path_matcher, None)
EnvVarLoader.add_constructor('!path', path_constructor)