$(document).ready(function() {
    $('.result-container').hide();

    // Display the current date
    const currentDate = new Date().toLocaleDateString();
    $('#current-date').text(currentDate);

    // Function to reset image previews
    function resetImagePreviews() {
        // Hide the preview containers and remove background images
        $('#imagePreview1').css('background-image', 'none').parent().hide();
        $('#imagePreview2').css('background-image', 'none').parent().hide();
    }

    // Function to reset the form for a new prediction
    function resetForm() {
        $('#upload-file')[0].reset();  // Reset the file input form
        resetImagePreviews();          // Clear the image previews
        $('#predict-btn').show();      // Show the predict button
        $('.result-container').hide(); // Hide the result container
    }

    // Reset form when the case-id input is changed
    $('#case-id').on('input', function() {
        resetForm();
    });
    
    // Function to preview the selected images
    function previewImage(input, previewElementId) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                $(`#${previewElementId}`).css('background-image', `url(${e.target.result})`);
                $(`#${previewElementId}`).parent().show(); // Show the preview container
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    // Show preview when an image is selected
    $('#image1').change(function() {
        previewImage(this, 'imagePreview1');
    });
    
    $('#image2').change(function() {
        previewImage(this, 'imagePreview2');
    });

    // Handle the "Predict!" button click
    $('#predict-btn').click(function(event) {
        event.preventDefault();  // Prevent default form submission
       
        // Capture the case ID
        const caseId = $('#case-id').val();

        // Check if both files are selected
        if (!$('#image1').val() || !$('#image2').val()) {
            alert('Please select both images before predicting.');
            return;
        }
        
        $('#predict-btn').hide();
        // Show loader and hide result
        $('.loader').show();
        
        // Prepare form data with the two images
        const formData = new FormData();
        formData.append('image1', $('#image1')[0].files[0]);
        formData.append('image2', $('#image2')[0].files[0]);
        formData.append('case_id', caseId);

        // Send AJAX request to /uploads endpoint
        $.ajax({
            url: '/uploads',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                console.log('Upload successful:', response)
                if (response.success) {
                    $('.loader').hide();
                    // Display results
                    $('#response-class').text((response.response_probability * 100).toFixed(2) + '%');
                    $('#non-response-class').text((response.non_response_probability * 100).toFixed(2) + '%');
                    $('#case-id-display').text(caseId);  // Show the case ID in the result
                    $('.result-container').show();
                } else {
                    alert('Prediction failed. Please try again.');
                }
            },
            error: function(error) {
                $('.loader').hide();
                alert('An unexpected error occurred. Please try again.');
                console.error('Error:', error);
            },
        });
    });
});
