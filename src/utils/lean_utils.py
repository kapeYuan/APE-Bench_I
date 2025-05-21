# Copyright (2025) Bytedance Ltd. and/or its affiliates.

def remove_lean_comments(src: str) -> str:
    i = 0
    n = len(src)
    out = []
    block_nest = 0
    while i < n:
        if block_nest == 0:
            if src.startswith("--", i):
                j = src.find("\n", i + 2)
                if j == -1:
                    break
                else:
                    out.append("\n")
                    i = j + 1
            elif src.startswith("/-", i):
                block_nest = 1
                i += 2
            else:
                out.append(src[i])
                i += 1
        else:
            if src.startswith("/-", i):
                block_nest += 1
                i += 2
            elif src.startswith("-/", i):
                block_nest -= 1
                i += 2
            else:
                i += 1

    return "".join(out)