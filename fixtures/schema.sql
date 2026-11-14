PRAGMA foreign_keys=ON;
CREATE TABLE customers(customer_id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL, email TEXT);
CREATE TABLE orders(order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id), amount_cents INTEGER NOT NULL CHECK(amount_cents>=0), note TEXT);
