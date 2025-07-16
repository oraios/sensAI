from sensai.util.string import ToStringMixin


class A(ToStringMixin):
    def __init__(self, p1):
        self.p1 = p1
        self.p2 = 2
        self.p3 = self
        self.p5 = ["foo", self]

    def _tostring_excludes(self):
        return ["p2"]


class B(ToStringMixin):
    def __init__(self, a):
        self.a = a


class Parent(ToStringMixin):
    def __init__(self, foo=10):
        self.foo = foo
        self.model = self.Child(self)

    class Child(ToStringMixin):
        def __init__(self, parent):
            self.parent = parent


def test_ToStringMixin_recursion():
    s = str(A("foo"))
    assert s == "A[p1='foo', p3=A[<<], p5=['foo', A[<<]]]"
    s = str(B(A("foo")))
    assert s == "B[a=A[p1='foo', p3=A[<<], p5=['foo', A[<<]]]]"
    s = str(Parent())
    assert s == "Parent[foo=10, model=Parent.Child[parent=Parent[<<]]]"
