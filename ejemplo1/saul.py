import tensorflow as tf

nodo1 = tf.constant(5)
nodo2 = tf.constant(3)
nodo3 = tf.add(nodo1, nodo2)

sess = tf.Session()

print("tf.add: ", sess.run(nodo3))

# segundo ejemplo

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + es lo mismo que si hicieramos tf.add(nodo1, nodo2)

print("adder_node: ", sess.run( adder_node, {a: 3, b: 4.5}))
# print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# hacemos que sea mas complejo

add_and_triple = adder_node * 3.
print("add_and_triple = adder_node * 3.: ", sess.run(add_and_triple, {a: 3, b: 4.5}))

# tercer ejemplo
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# print(sess.run(linear_model, {x: [1, 2, 3, 4]})) ?????

# y = tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
