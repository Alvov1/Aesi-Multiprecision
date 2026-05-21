import json
import sys


def make_summary(json_path: str) -> str:
    with open(json_path) as f:
        benchmarks = json.load(f)["benchmarks"]

    data: dict[str, dict[str, int]] = {}
    libraries: list[str] = []
    for b in benchmarks:
        op, lib = b["name"].split("_", 1)
        if op not in data:
            data[op] = {}
        data[op][lib] = int(b["real_time"])
        if lib not in libraries:
            libraries.append(lib)

    header = "| Operation | " + " | ".join(f"{lib} (ns)" for lib in libraries) + " |"
    sep    = "|-----------|" + "|".join("-----------:" for _ in libraries) + "|"
    rows   = [
        f"| {op} | " + " | ".join(str(vals.get(lib, "—")) for lib in libraries) + " |"
        for op, vals in data.items()
    ]

    return "\n".join([header, sep] + rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmarks.json>", file=sys.stderr)
        sys.exit(1)
    print(make_summary(sys.argv[1]))
