document.addEventListener('DOMContentLoaded', function() {
    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Image Preview
    const imageInput = document.getElementById('image');
    const imagePreview = document.getElementById('imagePreview');

    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            }
            reader.readAsDataURL(file);
        }
    });

    // Registration Form Submit
    const registrationForm = document.getElementById('registrationForm');
    registrationForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(registrationForm);
        
        try {
            const response = await fetch('/register_face', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('Registration successful!');
                registrationForm.reset();
                imagePreview.innerHTML = '';
            } else {
                alert(data.message || 'Registration failed!');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during registration');
        }
    });

    // Load Attendance Data
    function loadAttendanceData() {
        const date = document.getElementById('attendanceDate').value;
        const department = document.getElementById('filterDepartment').value;
        const semester = document.getElementById('filterSemester').value;

        fetch(`/get_attendance?date=${date}&department=${department}&semester=${semester}`)
            .then(response => response.json())
            .then(data => {
                const tableBody = document.getElementById('attendanceData');
                tableBody.innerHTML = '';

                data.forEach(record => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${record.usn}</td>
                        <td>${record.name}</td>
                        <td>${record.Positin}</td>
                        <td>${record.Year_of_Joing}</td>
                        <td>${record.mobile_number_and_email}</td>
                        <td>${record.time}</td>
                    `;
                    tableBody.appendChild(row);
                });
            })
            .catch(error => console.error('Error:', error));
    }

    // Attendance Filters Event Listeners
    document.getElementById('attendanceDate').addEventListener('change', loadAttendanceData);
    document.getElementById('filterDepartment').addEventListener('change', loadAttendanceData);
    document.getElementById('filterSemester').addEventListener('change', loadAttendanceData);

    // Show Success Animation
    window.showSuccessAnimation = function(studentName) {
        const overlay = document.getElementById('successOverlay');
        const studentNameElement = document.getElementById('markedStudentName');
        
        studentNameElement.textContent = studentName;
        overlay.style.display = 'flex';
        
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 3000);
    }

    // Initial load of attendance data
    loadAttendanceData();
}); 