import express from 'express';
import upload from '../middleware/upload.js';
import { uploadFile, getFiles, getFile } from '../controllers/fileController.js';

const router = express.Router();

router.post('/upload', upload.single('file'), uploadFile);
router.get('/', getFiles);
router.get('/:filename', getFile);

export default router;