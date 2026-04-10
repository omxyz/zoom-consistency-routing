"""Patch model_factory.py to add zoom_consistency_router."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "model_factory.py"

with open(path) as f:
    content = f.read()

insert_block = '''    elif model_type == "zoom_consistency_router":
        from models.zoom_consistency_router import ZoomConsistencyRouterModel
        model = ZoomConsistencyRouterModel()
        model.load_model()
'''

target = '    else:\n        raise ValueError'
if "zoom_consistency_router" not in content:
    content = content.replace(target, insert_block + target)
    with open(path, "w") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Already patched")
