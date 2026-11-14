SELECT o.order_id, c.region, p.payment_id, o.amount_cents
FROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN payments p ON p.order_id = o.order_id
WHERE c.active = 1 AND (o.amount_cents >= :min_amount OR o.priority = 'rush')
  AND p.status = :paid_status
