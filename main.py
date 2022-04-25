def solve(digits):
    # t = int("".join([str(i) for i in digits])) + 1
    return [i for i in str(int("".join([str(i) for i in digits])) + 1)]



if __name__ == "__main__":
    digits = [3, 9, 9]
    print(*solve(digits))