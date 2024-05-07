import ast
import pkg_resources

def extract_imports(path):
    with open(path, "r") as file:
        tree = ast.parse(file.read(), filename=path)
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])  # Only get the top-level package name
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.add(node.module.split('.')[0])  # Only get the top-level package name

    return imports

def get_installed_packages():
    installed_packages = {dist.project_name for dist in pkg_resources.working_set}
    return installed_packages

def main(script_path):
    imports = extract_imports(script_path)
    installed_packages = get_installed_packages()
    requirements = imports.intersection(installed_packages)

    with open("requirements.txt", "w") as file:
        for package in requirements:
            file.write(package + "\n")

if __name__ == "__main__":
    script_path = "app.py"  # Update this to the path of your Python script
    main(script_path)
