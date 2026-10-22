const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = process.env.PORT || 3000;
const DATA_DIR = path.join(__dirname, 'data');
const USERS_FILE = path.join(DATA_DIR, 'users.json');

// Garante que o diretório de dados e arquivo de usuários existam
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}
if (!fs.existsSync(USERS_FILE)) {
    fs.writeFileSync(USERS_FILE, JSON.stringify([], null, 2), 'utf8');
}

// MIME types para arquivos estáticos
const MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff2': 'font/woff2',
    '.woff': 'font/woff',
    '.ttf': 'font/ttf'
};

function readUsers() {
    try {
        const raw = fs.readFileSync(USERS_FILE, 'utf8');
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
        console.error('Erro ao ler users.json:', e.message);
        return [];
    }
}

function writeUsers(users) {
    fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2), 'utf8');
}

function sendJSON(res, statusCode, data) {
    res.writeHead(statusCode, {
        'Content-Type': 'application/json; charset=utf-8',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    });
    res.end(JSON.stringify(data));
}

const server = http.createServer((req, res) => {
    // CORS Preflight
    if (req.method === 'OPTIONS') {
        res.writeHead(204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        });
        return res.end();
    }

    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // ================= API ENDPOINTS =================

    // Status do servidor
    if (pathname === '/api/status' && req.method === 'GET') {
        const users = readUsers();
        return sendJSON(res, 200, {
            status: 'online',
            service: 'FaceAuth Local Server',
            totalRegisteredUsers: users.length,
            databasePath: USERS_FILE,
            uptime: Math.floor(process.uptime())
        });
    }

    // Listar todos os usuários cadastrados
    if (pathname === '/api/users' && req.method === 'GET') {
        const users = readUsers();
        return sendJSON(res, 200, {
            success: true,
            count: users.length,
            users
        });
    }

    // Cadastrar / atualizar usuário
    if (pathname === '/api/users' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
            try {
                const payload = JSON.parse(body);

                if (!payload.name || !Array.isArray(payload.descriptor) || payload.descriptor.length !== 128) {
                    return sendJSON(res, 400, {
                        success: false,
                        error: 'Parâmetros inválidos: o nome e o vetor de 128 dimensões do rosto são obrigatórios.'
                    });
                }

                const users = readUsers();
                const now = new Date().toISOString();
                const normalizedName = payload.name.trim();

                // Procura se já existe usuário com o mesmo nome para atualizar
                const existingIndex = users.findIndex(
                    u => u.name.toLowerCase() === normalizedName.toLowerCase()
                );

                const userData = {
                    id: existingIndex >= 0 ? users[existingIndex].id : 'user_' + Date.now(),
                    name: normalizedName,
                    descriptor: payload.descriptor,
                    templates: Array.isArray(payload.templates) ? payload.templates : [payload.descriptor],
                    registeredAt: existingIndex >= 0 ? users[existingIndex].registeredAt : now,
                    updatedAt: now
                };

                if (existingIndex >= 0) {
                    users[existingIndex] = userData;
                } else {
                    users.push(userData);
                }

                writeUsers(users);
                console.log(`[Servidor] Usuário "${userData.name}" cadastrado/atualizado com sucesso no servidor.`);

                return sendJSON(res, 200, {
                    success: true,
                    message: `Usuário "${userData.name}" salvo com sucesso no servidor!`,
                    user: userData
                });
            } catch (err) {
                return sendJSON(res, 400, {
                    success: false,
                    error: 'Corpo da requisição JSON inválido: ' + err.message
                });
            }
        });
        return;
    }

    // Excluir usuário específico por ID ou Nome
    if (pathname.startsWith('/api/users/') && req.method === 'DELETE') {
        const identifier = decodeURIComponent(pathname.replace('/api/users/', ''));
        let users = readUsers();
        const initialCount = users.length;

        users = users.filter(u => u.id !== identifier && u.name.toLowerCase() !== identifier.toLowerCase());

        if (users.length !== initialCount) {
            writeUsers(users);
            console.log(`[Servidor] Usuário "${identifier}" removido do servidor.`);
            return sendJSON(res, 200, {
                success: true,
                message: `Usuário "${identifier}" removido com sucesso.`
            });
        } else {
            return sendJSON(res, 404, {
                success: false,
                error: `Usuário "${identifier}" não encontrado no servidor.`
            });
        }
    }

    // Limpar todos os cadastros
    if (pathname === '/api/users' && req.method === 'DELETE') {
        writeUsers([]);
        console.log(`[Servidor] Todos os cadastros foram limpos do servidor.`);
        return sendJSON(res, 200, {
            success: true,
            message: 'Todos os registros foram removidos do servidor.'
        });
    }

    // ================= STATIC FILE SERVER =================
    let safePath = path.normalize(pathname).replace(/^(\.\.[\/\\])+/, '');
    if (safePath === '/' || safePath === '\\') safePath = 'index.html';

    const filePath = path.join(__dirname, safePath);
    const ext = path.extname(filePath).toLowerCase();

    fs.stat(filePath, (err, stats) => {
        if (err || !stats.isFile()) {
            res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
            return res.end('404 - Arquivo não encontrado');
        }

        const contentType = MIME_TYPES[ext] || 'application/octet-stream';
        res.writeHead(200, {
            'Content-Type': contentType,
            'Cache-Control': 'no-cache'
        });

        fs.createReadStream(filePath).pipe(res);
    });
});

server.listen(PORT, () => {
    console.log(`\n======================================================`);
    console.log(`   🚀 FaceAuth Server iniciado com sucesso!`);
    console.log(`   🌐 Endereço da Aplicação: http://localhost:${PORT}`);
    console.log(`   📂 Banco de Usuários:     ${USERS_FILE}`);
    console.log(`======================================================\n`);
});
