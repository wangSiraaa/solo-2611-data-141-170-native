CREATE TABLE customers(customer_id TEXT PRIMARY KEY, region TEXT, active INTEGER NOT NULL);
CREATE TABLE orders(order_id TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers, amount_cents INTEGER NOT NULL, priority TEXT);
CREATE TABLE payments(payment_id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders, status TEXT);
INSERT INTO customers VALUES ('C1','north',1),('C2','south',0),('C3',NULL,1);
INSERT INTO orders VALUES ('O1','C1',5000,'normal'),('O2','C2',1500,'normal'),('O3','C2',3500,'normal'),('O4','C1',100,'rush'),('O5',NULL,9000,'rush'),('O6','C3',4500,'normal');
INSERT INTO payments VALUES ('P11','O1','paid'),('P12','O1','paid'),('P21','O2','pending'),('P31','O3',NULL),('P61','O6','pending'),('P62','O6','paid');
