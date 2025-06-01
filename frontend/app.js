let map;
let markers = [];
const UKRAINE_CENTER = { lat: 48.3794, lng: 31.1656 };
const UKRAINE_ZOOM = 6;

// Initialize the map
function initMap() {
    console.log('initMap function called');
    console.log('Google Maps API available:', typeof google !== 'undefined');
    console.log('Map container exists:', !!document.getElementById('map'));
    
    try {
        console.log('Attempting to create map...');
        map = new google.maps.Map(document.getElementById('map'), {
            center: UKRAINE_CENTER,
            zoom: UKRAINE_ZOOM,
            styles: [
                {
                    "featureType": "administrative",
                    "elementType": "geometry",
                    "stylers": [{"visibility": "on"}]
                },
                {
                    "featureType": "landscape",
                    "elementType": "geometry",
                    "stylers": [{"color": "#f5f5f5"}]
                },
                {
                    "featureType": "water",
                    "elementType": "geometry",
                    "stylers": [{"color": "#e9e9e9"}]
                }
            ]
        });
        console.log('Map created successfully');
        console.log('Map object:', map);
    } catch (error) {
        console.error('Error initializing map:', error);
        console.error('Error stack:', error.stack);
    }
}

// Clear all markers from the map
function clearMarkers() {
    console.log('Clearing markers, current count:', markers.length);
    markers.forEach(marker => marker.setMap(null));
    markers = [];
}

// Add a marker for a city
function addCityMarker(cityName, lat, lng) {
    console.log('Adding marker for city:', cityName, 'at coordinates:', lat, lng);
    try {
        const marker = new google.maps.Marker({
            position: { lat, lng },
            map: map,
            title: cityName,
            animation: google.maps.Animation.DROP
        });

        const infoWindow = new google.maps.InfoWindow({
            content: `<div class="marker-info">${cityName}</div>`
        });

        marker.addListener('click', () => {
            infoWindow.open(map, marker);
        });

        markers.push(marker);
        console.log('Marker added successfully');
        return marker;
    } catch (error) {
        console.error('Error adding marker:', error);
        return null;
    }
}

// Extract city names from the AI response
function extractCities(text) {
    console.log('Extracting cities from text:', text);
    const cityPattern = /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g;
    const cities = text.match(cityPattern) || [];
    const uniqueCities = [...new Set(cities)];
    console.log('Extracted cities:', uniqueCities);
    return uniqueCities;
}

// City coordinates database (you should expand this)
const cityCoordinates = {
    'Donetsk': { lat: 48.0159, lng: 37.8028 },
    'Kharkiv': { lat: 49.9935, lng: 36.2304 },
    'Luhansk': { lat: 48.5740, lng: 39.3078 },
    'Kupiansk': { lat: 49.7063, lng: 37.6167 },
    'Pokrovsk': { lat: 48.2817, lng: 37.1758 },
    'Horlivka': { lat: 48.3000, lng: 38.0500 },
    'Vodiane': { lat: 47.4833, lng: 37.4833 },
    'Novovodiane': { lat: 47.5000, lng: 37.5000 },
    'Balka Zhuravka': { lat: 48.3167, lng: 37.2500 },
    'Novoiehorivka': { lat: 48.3167, lng: 37.2500 }
};

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    console.log('Map container exists:', !!document.getElementById('map'));
    console.log('Google Maps API available:', typeof google !== 'undefined');
    
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function addMessage(content, isUser = false) {
        console.log('Adding message:', { content, isUser });
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'system'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        if (isUser) {
            messageContent.textContent = content;
        } else {
            // Render markdown as HTML for system messages
            messageContent.innerHTML = marked.parse(content);
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // If it's a system message, highlight cities on the map
        if (!isUser && map) {
            console.log('Processing cities for map highlighting');
            const cities = extractCities(content);
            clearMarkers();
            cities.forEach(city => {
                if (cityCoordinates[city]) {
                    console.log('Adding marker for city:', city);
                    addCityMarker(city, cityCoordinates[city].lat, cityCoordinates[city].lng);
                } else {
                    console.log('No coordinates found for city:', city);
                }
            });
        } else if (!isUser && !map) {
            console.warn('Map not initialized, cannot highlight cities');
        }
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        console.log('Sending message:', message);
        addMessage(message, true);
        userInput.value = '';

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message system';
        loadingDiv.innerHTML = '<div class="message-content"><div class="loading"></div></div>';
        chatMessages.appendChild(loadingDiv);

        try {
            console.log('Making API request to backend');
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            console.log('Received response from backend:', data);
            
            chatMessages.removeChild(loadingDiv);
            addMessage(data.response);
        } catch (error) {
            console.error('Error in sendMessage:', error);
            chatMessages.removeChild(loadingDiv);
            addMessage('Sorry, there was an error processing your request. Please try again.');
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

window.initMap = initMap; 