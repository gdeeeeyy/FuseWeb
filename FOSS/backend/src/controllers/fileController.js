import Model from '../models/Model.js';
import path from 'path';
import fs from 'fs';

export const uploadFile = async (req, res) => {
    try {
        const newModel = await Model.create({ filename: req.file.filename, filepath: req.file.path });
        res.json({ message: 'File uploaded successfully', file: req.file });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};

export const getFiles = async (req, res) => {
    try {
        const models = await Model.findAll();
        res.json(models);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};

export const getFile = (req, res) => {
    const filePath = path.join(__dirname, '../uploads', req.params.filename);
    if (fs.existsSync(filePath)) {
        res.sendFile(filePath);
    } else {
        res.status(404).json({ error: 'File not found' });
    }
};
