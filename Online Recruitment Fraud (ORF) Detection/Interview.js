   const express = require('express');
   const axios = require('axios');
   const cors = require('cors');
   const app = express();
   const PORT = process.env.PORT || 5000;

   app.use(cors());
   app.use(express.json());

   let searches = [];

   app.get('/api/weather', async (req, res) => {
     const city = req.query.city;
     const response = await axios.get(`API_URL?city=${city}&appid=YOUR_API_KEY`);
     res.json(response.data);
   });

   app.get('/api/weather/current', async (req, res) => {
     const { lat, lon } = req.query;
     const response = await axios.get(`API_URL?lat=${lat}&lon=${lon}&appid=YOUR_API_KEY`);
     res.json(response.data);
   });

   app.post('/api/search', (req, res) => {
     const { city } = req.body;
     searches.push(city);
     res.status(201).send();
   });

   app.get('/api/searches', (req, res) => {
     res.json(searches);
   });

   app.listen(PORT, () => {
     console.log(`Server is running on http://localhost:${PORT}`);
   });
   