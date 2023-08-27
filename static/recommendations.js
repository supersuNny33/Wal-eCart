// Add event listener to "Add to List" buttons
const addToListButtons = document.querySelectorAll('.add-to-list-btn');
addToListButtons.forEach(button => {
    button.addEventListener('click', function() {
        const productId = button.getAttribute('data-product-id');
        addToShoppingList(productId);
    });
});

// Function to add product to shopping list
function addToShoppingList(productId) {
    // Send an AJAX request to your server to add the product to the shopping list
    // You can use libraries like jQuery or fetch API for this
    // Example using fetch API:
    fetch(`/add_to_list`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ product_id: productId }),
    })
        .then(response => response.json())
        .then(data => {
            // Handle success or failure
            if (data.success) {
                // Product added successfully, you can provide feedback to the user
            } else {
                // Product couldn't be added, show an error message
            }
        })
        .catch(error => {
            // Handle error
            console.error('Error:', error);
        });
}
