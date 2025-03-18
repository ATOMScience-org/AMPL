"""An example file that produces a custom parameters for the LULESH example."""

from maestrowf.datastructures.core import ParameterGenerator
from atomsci.ddm.utils import hyperparam_search_wrapper as hsw

#---------------------------------------------------------------------
def get_custom_generator(env, **kwargs):
    """Create a custom populated ParameterGenerator.

    This function recreates the exact same parameter set as the sample LULESH
    specifications. The point of this file is to present an example of how to
    generate custom parameters.

    Returns:
        A ParameterGenerator populated with parameters.
    """

    nn_ecfp_pparams = hsw.parse_params(['--config', env.find('JSON_FILE').value])
    commands = hsw.build_search(nn_ecfp_pparams).generate_maestro_commands()

    p_gen = ParameterGenerator()
    params = {
        "COMMAND": {
            "values": [command for command in commands],
            "label": ["COMMAND.%s"%str(i) for i in range(len(commands))]
        },
    }

    for key, value in params.items():
        print(value)
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen

