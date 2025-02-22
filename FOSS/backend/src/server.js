import express from 'express';
import cors from 'cors';
import { Sequelize } from 'sequelize';
import fileRoutes from './routes/fileRoutes.js';
import Model from './models/Model.js';

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use('/api/files', fileRoutes);

// PostgreSQL Connection
const sequelize = new Sequelize('3dmodels', 'username', 'password', {
    host: 'localhost',
    dialect: 'postgres'
});

sequelize.authenticate()
    .then(() => console.log('PostgreSQL Connected'))
    .catch(err => console.log('Error: ' + err));

// Sync Database
sequelize.sync();

// Start Server
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));