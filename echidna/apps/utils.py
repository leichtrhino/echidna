
import yaml
import json
import csv
import argparse
import jsonschema

from ..data.samples import Datasource

def make_config_action(json_schema : dict=None):
    class ConfigAction(argparse.Action):
        def __call__(self,
                     parser : argparse.ArgumentParser,
                     namespace : argparse.Namespace,
                     values,
                     option_string=None):
            config_path = values
            if config_path.endswith('.yaml') \
               or config_path.endswith('.yml'):
                with open(config_path, 'r') as fp:
                    config = yaml.load(fp, yaml.Loader)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as fp:
                    config = json.load(fp)
            else:
                with open(config_path, 'r') as fp:
                    config = yaml.load(fp, yaml.Loader)

            if json_schema is not None:
                jsonschema.validate(instance=config, schema=json_schema)

            for k, v in config.items():
                action = parser._option_string_actions\
                               .get(f'--{k.replace("_", "-")}', None)
                if action is not None:
                    action(parser, namespace, v, option_string=k)
                else:
                    setattr(namespace, k, v)

        def format_usage():
            return 'sample format usage'

    return ConfigAction

def make_structure_action(structure_class, json_schema=None):
    class StructureAction(argparse.Action):
        def __call__(self,
                     parser : argparse.ArgumentParser,
                     namespace : argparse.Namespace,
                     values,
                     option_string=None):
            if type(values) == str:
                values = json.loads(values)
            if json_schema:
                jsonschema.validate(instance=values, schema=json_schema)
            values = structure_class.from_dict(values)

            setattr(namespace, self.dest, values)

    return StructureAction

class DatasourceAction(argparse.Action):
    def __call__(self,
                 parser : argparse.ArgumentParser,
                 namespace : argparse.Namespace,
                 values,
                 option_string=None):

        datasource_path = values
        if datasource_path.endswith('.csv'):
            with open(datasource_path, 'r') as fp:
                csv_reader = csv.reader(fp)
                h = next(csv_reader)
                datasource = [
                    dict(zip(h, (c or None for c in r)))
                    for r in csv_reader
                ]
        elif datasource_path.endswith('.json'):
            with open(datasource_path, 'r') as fp:
                datasource = json.load(fp)
        else:
            with open(datasource_path, 'r') as fp:
                datasource = yaml.load(fp, yaml.Loader)

        datasource = [Datasource.from_dict(d) for d in datasource]
        setattr(namespace, self.dest, datasource)

