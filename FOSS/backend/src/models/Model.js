import { Sequelize, DataTypes } from 'sequelize';
import sequelize from '../server.js';

const Model = sequelize.define('Model', {
    filename: {
        type: DataTypes.STRING,
        allowNull: false
    },
    filepath: {
        type: DataTypes.STRING,
        allowNull: false
    },
    uploadedAt: {
        type: DataTypes.DATE,
        defaultValue: Sequelize.NOW
    }
});

export default Model;