"""Step registry and pipelines.

There is one analysis pipeline for every part. Each step declares when it applies (`when`), from what earlier steps
measured, e.g. the bore and bolt-circle steps run only when the outline step found a round part, the generic hole
finder runs on every part, and patterns analyses holes from all sources that ran.

Add a detector by creating a module in partvision/steps/ with a function decorated by @step(...). It is imported
automatically. Add it to a pipeline below, or run any list with --steps.
"""
STEPS = {}

# Ordered step lists. Steps whose `when` is false for the part at hand are skipped (and listed as skipped).
PIPELINES = {
    "inspect": ["part", "outline", "bore", "round_rim", "bolt_circle", "rim_notches", "rings", "holes", "patterns",
                "pockets", "scale", "ratios"],
    "drawing": ["part", "drawing_load", "drawing_align", "drawing_outline", "drawing_features", "drawing_report"],
}


def step(name, needs=(), when=None):
    """Register fn(ctx) -> dict as a pipeline step.
    needs: steps that must have run first. when: fn(ctx) -> bool, whether the step applies to this part
    (None = always). A step whose needed step was skipped is skipped too."""
    def deco(fn):
        STEPS[name] = dict(fn=fn, needs=tuple(needs), when=when, doc=(fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return deco


def run(ctx, names):
    for name in names:
        if name in ctx.results or name in ctx.skipped:
            continue
        if name not in STEPS:
            raise SystemExit(f"unknown step '{name}'. Available: {', '.join(sorted(STEPS))}")
        s = STEPS[name]
        if any(n in ctx.skipped for n in s["needs"]):
            ctx.skipped.append(name)
            continue
        missing = [n for n in s["needs"] if n not in ctx.results]
        if missing:
            raise SystemExit(f"step '{name}' needs {', '.join(missing)} to run before it")
        if s["when"] is not None and not s["when"](ctx):
            ctx.skipped.append(name)
            continue
        out = s["fn"](ctx)
        ctx.results[name] = out if out is not None else {}
