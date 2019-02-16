import sys, os
import json
import configparser


class AppHelper(object):
    """
    This class is a helper class. It provides secrets so that I can use a gitignore. 
    """

    def __init__(self, config_file="app_config.conf", env="AZ_AUTO_ML"):
        self.subscription_id = None
        self.aml_workspace_name = None
        self.aml_resource_group = None
        self.aml_location = None
        self.username = None
        self.password = None
        self.tenant_id = None
        self.set_config(config_file, env)

    def set_config(self, config_file,  env):
        """
        Sets configuration variables for the application
        :param config_file: the path to the configuration file
        :param env: the environment string to parse in config file
        :return None
        """
        config = configparser.RawConfigParser(allow_no_value=True)
        config.read(filenames = [config_file])
            
        ### Setting values here ###
        self.subscription_id = config.get(env, "SUBSCRIPTION_ID")
        self.aml_workspace_name = config.get(env, "AML_WORKSPACE_NAME")
        self.aml_resource_group = config.get(env, "AML_RESOURCE_GROUP")
        self.aml_location = config.get(env, "AML_LOCATION")
        self.username = config.get(env, "USERNAME")
        self.password = config.get(env, "PASSWORD")
        self.tenant_id = config.get(env, "TENANT_ID")
