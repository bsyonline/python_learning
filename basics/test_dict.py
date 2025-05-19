# 字典定义
d = {"name": "zhangsan", "age": 20}
d1 = dict(name="zhangsan", age=20)
d2 = dict([("name", "zhangsan"), ("age", 20)])
d3 = dict({"name": "zhangsan", "age": 20})
d4 = dict(zip(["name", "age"], ["zhangsan", 20]))
d5 = {k: v for k, v in [("name", "zhangsan"), ("age", 20)]}
d6 = {k: v for k, v in {"name": "zhangsan", "age": 20}.items()}
d7 = {("name", "age")[i]: ("zhangsan", 20)[i] for i in range(2)}
d8 = dict.fromkeys(["name", "age"], ["zhangsan", 20])
print(d)  # {'name': 'zhangsan', 'age': 20}
print(type(d))  # <class 'dict'>

# 通过键可以获取字典中对应的值
print(d["name"])  # zhangsan
print(d.get("age"))  # 20
print(d.get("age"), 0)  # 20

# 通过键可以修改字典中对应的值
d["gender"] = 'male'
print(d)  # {'name': 'zhangsan', 'age': 20
d = {}
d['name'] = "lisi"
d['age'] = 20
print(d)  # {'name': 'lisi', 'age': 20}
d['age'] = 21
print(d)  # {'name': 'lisi', 'age': 21}
d.update(gender='female', age=22)
print(d)  # {'name': 'lisi', 'age': 22, 'gender': 'female'}

# 删除字典中的元素
d.popitem()
print(d)  # {'name': 'lisi', 'age': 22}
d.pop('age')
print(d)  # {'name': 'lisi'}
del d['name']
print(d)  # {}
d.clear()
print(d)  # {}

print(len(d))

# 对字典进行遍历
for e in d1:
    print(e.title() + ":" + str(d1[e]))

# 遍历字典的键
for k in d1.keys():
    print(k)

# 遍历字典的值
for v in d1.values():
    print(v)

for k, v in d1.items():
    print(k.title() + ":" + str(v))

# **kwargs可以传递任意数量的键值对
def connect(fn, **kwargs):
    print(type(kwargs))
    print(kwargs)

connect(fn='connect', server='localhost', port=3306, user='root', password='Py1hon!Xt')

# 将字典传递给函数，则需要**
config = {'server': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'Py1thon!Xt12'}

connect(fn='connect', **config)