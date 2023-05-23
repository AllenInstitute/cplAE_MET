def load_config(config_file=None, verbose=False):
    """Loads dictionary with path names and any other variables set through config.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """
    import toml
    from pathlib import Path
    package_dir = Path(__file__).parent.parent.parent.absolute()
    config_file = package_dir / config_file
    if not config_file:
        config_file = package_dir / "config.toml"

    if not Path(config_file).is_file():
        print(f'Did not find config file: {config_file}')

    config = toml.load(config_file)
    config.update({'package_dir': package_dir, 'config_file': config_file})
    for key in config:
        if Path(config[key]).exists():
            config[key] = Path(config[key])
    if verbose:
        for key in config.keys():
            print(f'{key}: {config[key]}')
    return config, config_file
