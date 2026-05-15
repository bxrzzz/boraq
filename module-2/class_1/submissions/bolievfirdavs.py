def data_types():
    var1 = 42
    var2 = "Hello"
    var3 = 3.14
    var4 = True
    var5 = [1, 2, 3]
    var6 = {"key": "value"}
    var7 = (1, 2, 3)
    var8 = {1, 2, 3}

    variables = [var1, var2, var3, var4, var5, var6, var7, var8]
    types = [type(var).__name__ for var in variables]
    print(types)

if __name__ == '__main__':
    data_types()
