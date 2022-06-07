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

# Get length of argument.
def len_constructor(loader: yaml.Loader, node):

    # Node value is a string.
    if isinstance(node.value, str):
        is_number = False
        try:
            float(node.value)
            is_number = True
        except:
            pass
        if is_number:
            raise ValueError(f"cannot get length of an integer")
        else:
            return len(node.value)

    # Node value is a list of items.
    elif isinstance(node.value, list):

        # List contains single element.
        if len(node.value) == 1:

            # Element is scalar, so get length of that scalar.
            if isinstance(node.value[0], yaml.ScalarNode):
                return len(node.value[0].value)

            # Element is a sequence, so construct the sequence and return its length.
            elif isinstance(node.value[0], yaml.SequenceNode):
                return len(loader.construct_sequence(node.value[0], deep=True))

            # Unsupported node type.
            else:
                raise ValueError(f"unsupported node for length tag {node.value[0]}")
        else:
            return len(node.value)
    else:
        raise ValueError(f"unsupported node for length tag {node.value}")

# Configuration loader with custom tags.
class ConfigLoader(yaml.SafeLoader):
    pass
ConfigLoader.add_implicit_resolver('!path', path_matcher, None)
ConfigLoader.add_constructor('!path', path_constructor)
ConfigLoader.add_constructor('!join', join_constructor)
ConfigLoader.add_constructor('!len', len_constructor)