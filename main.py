import numpy as np
import tensorflow as tf


x_real = np.random.randn(2, 3)
x_imag = np.random.randn(2, 3)
x = x_real + 1j * x_imag

y = tf.cast(x, tf.int32)
print(y)
print(x)

isinstance()


https://etrans15.huawei.com/valid.aspx?d=GWmaNN/KrsqXmW1aCxoGiCuEZgbC5jZC1o4chA62Ulhlcl4aguw+Kv0bDh+LN7YB9F4QyQfahwEW+dLQRZISEh/payZ/0O/ChC+m0ajpMX3b07+Fuv2NWKpAXJM/IjEj74pu17kYAxym9eUXa82eBKNHUfNw3vneFGkeKYKkSB/YvXlbTHRm8Uqs0Y+jw2QPPrwkj5KxVFOzXem7jTZ9GIAcWUY0GmlJkpsRV1azP04nQUGmIWc8IJhLHZ6BIG9A1G09whNw4a//D2Bq7kc7EYsUp3H0ViEHmM+CSMSAwoZxsh2te5OO4ro8ONEjpt7dRUiVswh6vmbjpBkaXIRyhQ==