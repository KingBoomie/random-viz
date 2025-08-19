import inspect, ast

def dump_pydantic(obj):
    frame = inspect.currentframe().f_back
    src = inspect.getframeinfo(frame).code_context[0]
    expr = ast.parse(src).body[0].value

    if not (isinstance(expr, ast.Call) and
            isinstance(expr.func, ast.Name) and
            expr.func.id == 'dump_pydantic' and
            len(expr.args) == 1 and
            isinstance(expr.args[0], ast.Name)):
        raise ValueError("call must be dump_pydantic(varname) with a plain variable")

    name = expr.args[0].id
    with open(f"{name}.ndjson", "a") as f:
        f.write(obj.model_dump_json() + "\n")
