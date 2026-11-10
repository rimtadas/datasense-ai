ECOMMERCE_CONFIG = {
    "dataset_description": (
        "Synthetic e-commerce transaction dataset."
    ),
    "semantic_types": {
        "transaction_id": "identifier",
        "user_id": "identifier",
        "product_id": "identifier",
        "purchase_date": "datetime",
        "unit_price": "monetary",
        "subtotal": "monetary",
        "discount_amount": "monetary",
        "total_amount": "monetary",
    },
    "currency_column": "currency",
    "column_descriptions": {
        "total_amount": "Final transaction amount.",
        "quantity": "Number of product units purchased.",
        "payment_status": "Status of the transaction payment.",
        "country": "Country associated with the transaction.",
        "purchase_date": "Date and time of the purchase.",
    },
    "exclude_from_context": [
        "shipping_address"
    ],
}

ECOMMERCE_DATETIME_COLUMNS = [
    "purchase_date"
]