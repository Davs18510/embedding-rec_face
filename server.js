require('dotenv').config();
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const { createClient } = require('@supabase/supabase-js');
const { Pool } = require('pg');

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

// Gera vetor facial simulado de 128 dimensões com norma Euclidiana = 1.0
function generateSimulatedEmbedding(seed = Math.random()) {
    const raw = [];
    let norm = 0;
    for (let i = 0; i < 128; i++) {
        const val = Math.sin(seed * (i + 1) * 997.13) * 0.5 + Math.cos((i + 7) * 43.17) * 0.3;
        raw.push(val);
        norm += val * val;
    }
    norm = Math.sqrt(norm);
    return raw.map(v => Number((v / norm).toFixed(6)));
}

// Permite comunicação HTTPS segura mesmo em ambientes com proxy corporativo/inspeção SSL
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

// ===================== SUPABASE INTEGRATION =====================
let supabase = null;
let supabaseConfigured = false;
let supabaseTableReady = false;
let supabaseError = null;

function getSupabaseCredentials() {
    let supabaseUrl = process.env.SUPABASE_URL ||
        process.env.SUBASEBASE_URL ||
        process.env.NEXT_PUBLIC_SUPABASE_URL ||
        process.env.VITE_SUPABASE_URL ||
        '';

    let supabaseKey = process.env.SUPABASE_SECRET_KEY ||
        process.env.SUPABASE_SERVICE_ROLE_KEY ||
        process.env.SUPABASE_KEY ||
        process.env.SUPABASE_ANON_KEY ||
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ||
        '';

    const databaseUrl = process.env.DATABASE_URL ||
        process.env.POSTGRES_URL ||
        process.env.SUPABASE_DB_URL ||
        '';

    // Normaliza URL removendo /rest/v1 caso tenha sido colada junto
    supabaseUrl = supabaseUrl.trim().replace(/\/rest\/v1\/?$/, '').replace(/\/+$/, '');
    supabaseKey = supabaseKey.trim();

    return { supabaseUrl, supabaseKey, databaseUrl: databaseUrl.trim() };
}

async function initSupabase() {
    const { supabaseUrl, supabaseKey, databaseUrl } = getSupabaseCredentials();

    if (!supabaseUrl || !supabaseKey) {
        supabaseConfigured = false;
        supabaseError = 'Credenciais não encontradas no arquivo .env (preencha SUPABASE_URL e SUPABASE_KEY)';
        console.log(`\n[Supabase] ⚠️  ${supabaseError}`);
        return false;
    }

    try {
        supabase = createClient(supabaseUrl, supabaseKey, {
            auth: { persistSession: false, autoRefreshToken: false }
        });
        supabaseConfigured = true;
        supabaseError = null;
        console.log(`[Supabase] 🔌 Conectando ao projeto: ${supabaseUrl}`);

        // 1. Se DATABASE_URL foi configurado, garante schema via conexão PostgreSQL direta
        if (databaseUrl) {
            await ensureTableViaPostgres(databaseUrl);
        }

        // 2. Testa conexão e verifica se a tabela face_embeddings responde
        await checkAndEnsureTableStatus();

        // 3. Envia automaticamente os cadastros pré-existentes do servidor local
        await syncAllExistingUsersToSupabase();

        return true;
    } catch (err) {
        supabaseConfigured = false;
        supabaseError = err.message;
        console.error('[Supabase] ❌ Erro ao inicializar cliente Supabase:', err.message);
        return false;
    }
}

async function ensureTableViaPostgres(connectionString) {
    try {
        const pool = new Pool({
            connectionString,
            ssl: { rejectUnauthorized: false }
        });

        await pool.query(`
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding JSONB NOT NULL,
                timestamp BIGINT NOT NULL,
                registered_at TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_face_embeddings_name ON face_embeddings (name);
            CREATE INDEX IF NOT EXISTS idx_face_embeddings_timestamp ON face_embeddings (timestamp);
        `);

        await pool.end();
        supabaseTableReady = true;
        console.log('[Supabase] ✅ Tabela "face_embeddings" verificada/criada com sucesso no PostgreSQL!');
    } catch (err) {
        console.warn('[Supabase] Aviso ao tentar DDL direto via PostgreSQL:', err.message);
    }
}

async function checkAndEnsureTableStatus() {
    if (!supabase) return false;

    try {
        // Tenta consultar 1 registro para validar se a tabela já existe
        const { data, error } = await supabase
            .from('face_embeddings')
            .select('id')
            .limit(1);

        if (error) {
            // Se a tabela não existe
            if (error.code === '42P01' || error.message.includes('does not exist') || error.message.includes('not found') || error.code === 'PGRST205') {
                supabaseTableReady = false;
                console.log('[Supabase] ℹ️  A tabela "face_embeddings" ainda não existe na sua base.');
                console.log('[Supabase] 💡 Tentando criação automática via RPC / SQL...');

                // Tenta criar via RPC exec_sql se configurado
                try {
                    const rpcRes = await supabase.rpc('exec_sql', {
                        query: `
                            CREATE TABLE IF NOT EXISTS face_embeddings (
                                id TEXT PRIMARY KEY,
                                name TEXT NOT NULL,
                                embedding JSONB NOT NULL,
                                timestamp BIGINT NOT NULL,
                                registered_at TEXT,
                                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                            );
                        `
                    });
                    if (rpcRes && rpcRes.error) {
                        throw new Error(rpcRes.error.message);
                    }
                    supabaseTableReady = true;
                    console.log('[Supabase] ✅ Tabela "face_embeddings" criada com sucesso via RPC!');
                    return true;
                } catch (rpcErr) {
                    supabaseError = 'A tabela "face_embeddings" precisa ser criada no SQL Editor do Supabase (ou via DATABASE_URL no .env).';
                    console.log(`[Supabase] 💡 ${supabaseError}`);
                }
            } else {
                supabaseError = error.message;
            }
            return false;
        }

        supabaseTableReady = true;
        supabaseError = null;
        console.log('[Supabase] ✅ Tabela "face_embeddings" conectada e pronta para inserções!');
        return true;
    } catch (e) {
        supabaseTableReady = false;
        supabaseError = e.message;
        return false;
    }
}

// IDs que falharam e devem ser reenviados automaticamente
const pendingSupabaseSync = new Set();
const SUPABASE_SYNC_ATTEMPTS = 3;

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function sendUserToSupabase(user, maxAttempts = SUPABASE_SYNC_ATTEMPTS) {
    if (!supabase) {
        if (user && user.id) pendingSupabaseSync.add(user.id);
        return { success: false, error: 'Supabase não inicializado. Verifique o arquivo .env.' };
    }

    const payload = {
        id: user.id,
        name: user.name,
        embedding: user.descriptor, // Array de 128 números decimais
        timestamp: user.timestamp || Date.now(),
        registered_at: user.registeredAt || new Date().toISOString(),
        updated_at: new Date().toISOString()
    };

    let lastError = null;

    try {
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const { error } = await supabase
                .from('face_embeddings')
                .upsert(payload, { onConflict: 'id' });

            if (!error) {
                pendingSupabaseSync.delete(user.id);
                console.log(`[Supabase] 🚀 ✓ Usuário "${user.name}" (ID: ${user.id}) sincronizado automaticamente na base de dados!`);
                return { success: true };
            }

            lastError = error;
            console.error(`[Supabase] ❌ Tentativa ${attempt}/${maxAttempts} falhou para "${user.name}":`, error.message);

            if (attempt < maxAttempts) {
                await delay(400 * attempt);
            }
        }

        pendingSupabaseSync.add(user.id);
        return { success: false, error: lastError ? lastError.message : 'Erro desconhecido' };
    } catch (err) {
        pendingSupabaseSync.add(user.id);
        console.error(`[Supabase] Exceção ao sincronizar "${user.name}":`, err.message);
        return { success: false, error: err.message };
    }
}

// Reenvia periodicamente os registros que falharam (sem bloquear requisições)
let isRetryingPendingSync = false;

async function retryPendingSupabaseSyncs() {
    if (!supabase || !supabaseConfigured || pendingSupabaseSync.size === 0 || isRetryingPendingSync) return;

    isRetryingPendingSync = true;
    try {
        const users = readUsers();
        const batch = Array.from(pendingSupabaseSync).slice(0, 10);

        await Promise.all(batch.map(async (id) => {
            const user = users.find(u => u.id === id);
            if (!user) {
                pendingSupabaseSync.delete(id);
                return;
            }
            await sendUserToSupabase(user, 1);
        }));
    } finally {
        isRetryingPendingSync = false;
    }
}

setInterval(retryPendingSupabaseSyncs, 30000).unref();

async function syncAllExistingUsersToSupabase() {
    if (!supabase) return { synced: 0, total: 0 };

    const users = readUsers();
    if (users.length === 0) {
        console.log('[Supabase] Nenhum usuário local encontrado no servidor para envio.');
        return { synced: 0, total: 0 };
    }

    console.log(`[Supabase] 📦 Sincronizando ${users.length} usuário(s) pré-existente(s) com a base no Supabase...`);
    let successCount = 0;

    for (const u of users) {
        // Uma tentativa por usuário no bulk; falhas entram na fila de retentativa automática
        const result = await sendUserToSupabase(u, 1);
        if (result.success) successCount++;
    }

    console.log(`[Supabase] 🎉 Sincronização inicial concluída: ${successCount}/${users.length} usuários enviados ao Supabase.`);
    return { synced: successCount, total: users.length };
}

// ===================== HTTP SERVER =====================
const server = http.createServer(async (req, res) => {
    // CORS Preflight
    if (req.method === 'OPTIONS') {
        res.writeHead(204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        });
        return res.end();
    }

    const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    const pathname = parsedUrl.pathname;

    // ================= API ENDPOINTS =================

    // Status do servidor
    if (pathname === '/api/status' && req.method === 'GET') {
        const users = readUsers();
        const { supabaseUrl } = getSupabaseCredentials();

        return sendJSON(res, 200, {
            status: 'online',
            service: 'FaceAuth Local Server',
            totalRegisteredUsers: users.length,
            databasePath: USERS_FILE,
            supabase: {
                configured: supabaseConfigured,
                tableReady: supabaseTableReady,
                url: supabaseUrl ? supabaseUrl.replace(/https:\/\/(.*)\.supabase\.co.*/, '$1.supabase.co') : null,
                error: supabaseError
            },
            uptime: Math.floor(process.uptime())
        });
    }

    // Status detalhado do Supabase
    if (pathname === '/api/supabase/status' && req.method === 'GET') {
        // Tenta reavaliar caso o usuário tenha acabado de salvar o .env
        if (!supabaseConfigured) {
            require('dotenv').config({ override: true });
            await initSupabase();
        } else {
            await checkAndEnsureTableStatus();
        }

        const { supabaseUrl } = getSupabaseCredentials();
        return sendJSON(res, 200, {
            configured: supabaseConfigured,
            tableReady: supabaseTableReady,
            url: supabaseUrl || null,
            error: supabaseError
        });
    }

    // Forçar sincronização de todos os usuários com o Supabase
    if (pathname === '/api/supabase/sync-all' && req.method === 'POST') {
        if (!supabaseConfigured) {
            require('dotenv').config({ override: true });
            await initSupabase();
        }

        const result = await syncAllExistingUsersToSupabase();
        return sendJSON(res, 200, {
            success: true,
            synced: result.synced,
            total: result.total,
            message: `${result.synced}/${result.total} usuário(s) sincronizado(s) com o Supabase.`
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

    // Endpoint para simular cadastro de usuário com embedding 128D realista
    if (pathname === '/api/test/simulate-user' && req.method === 'POST') {
        const names = [
            'Lucas Rocha (Simulação)',
            'Mariana Costa (Simulação)',
            'Felipe Andrade (Simulação)',
            'Beatriz Lima (Simulação)'
        ];
        const randomName = names[Math.floor(Math.random() * names.length)];
        const simulatedEmbedding = generateSimulatedEmbedding(Date.now());
        const simId = 'usr_sim_' + Date.now().toString(36) + '_' + Math.random().toString(36).substring(2, 6);
        const now = new Date().toISOString();
        const nowTimestamp = Date.now();

        const newUser = {
            id: simId,
            name: randomName,
            descriptor: simulatedEmbedding,
            templates: [simulatedEmbedding],
            timestamp: nowTimestamp,
            registeredAt: now
        };

        const users = readUsers();
        users.push(newUser);
        writeUsers(users);

        // Envia automaticamente para o Supabase
        const supabaseResult = await sendUserToSupabase(newUser);

        return sendJSON(res, 201, {
            success: true,
            message: `Usuário simulado "${newUser.name}" gravado com sucesso!`,
            user: newUser,
            supabaseSync: supabaseResult
        });
    }

    // Cadastrar / atualizar usuário
    if (pathname === '/api/users' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', async () => {
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
                const nowTimestamp = Date.now();
                const normalizedName = payload.name.trim();

                // Procura se já existe usuário com o mesmo nome ou ID para atualizar
                const existingIndex = users.findIndex(
                    u => (payload.id && u.id === payload.id) || u.name.toLowerCase() === normalizedName.toLowerCase()
                );

                const finalId = payload.id || (existingIndex >= 0 ? users[existingIndex].id : 'usr_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8));
                const finalTimestamp = payload.timestamp || (existingIndex >= 0 && users[existingIndex].timestamp ? users[existingIndex].timestamp : nowTimestamp);

                const userData = {
                    id: finalId,
                    name: normalizedName,
                    descriptor: payload.descriptor,
                    templates: Array.isArray(payload.templates) ? payload.templates : [payload.descriptor],
                    timestamp: finalTimestamp,
                    registeredAt: existingIndex >= 0 && users[existingIndex].registeredAt ? users[existingIndex].registeredAt : now,
                    updatedAt: now
                };

                if (existingIndex >= 0) {
                    users[existingIndex] = userData;
                } else {
                    users.push(userData);
                }

                writeUsers(users);
                console.log(`[Servidor] Usuário "${userData.name}" (ID: ${userData.id}) salvo localmente.`);

                // Sincronização automática com Supabase
                if (!supabaseConfigured) {
                    // Tenta recarregar .env caso recém-salvo
                    require('dotenv').config({ override: true });
                    await initSupabase();
                }

                const supabaseSyncResult = await sendUserToSupabase(userData);

                return sendJSON(res, 200, {
                    success: true,
                    message: `Usuário "${userData.name}" cadastrado com sucesso!`,
                    user: userData,
                    supabaseSync: supabaseSyncResult
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

        const targetUser = users.find(u => u.id === identifier || u.name.toLowerCase() === identifier.toLowerCase());
        users = users.filter(u => u.id !== identifier && u.name.toLowerCase() !== identifier.toLowerCase());

        if (users.length !== initialCount) {
            writeUsers(users);
            console.log(`[Servidor] Usuário "${identifier}" removido do servidor local.`);

            // Remove do Supabase também
            if (supabase) {
                try {
                    const deleteId = targetUser ? targetUser.id : identifier;
                    await supabase.from('face_embeddings').delete().eq('id', deleteId);
                    console.log(`[Supabase] Registro "${deleteId}" removido da base Supabase.`);
                } catch (delErr) {
                    console.warn('[Supabase] Erro ao remover registro:', delErr.message);
                }
            }

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
        console.log(`[Servidor] Todos os cadastros locais foram limpos.`);

        return sendJSON(res, 200, {
            success: true,
            message: 'Todos os registros locais foram removidos.'
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

server.listen(PORT, async () => {
    console.log(`\n======================================================`);
    console.log(`   🚀 FaceAuth Server iniciado com sucesso!`);
    console.log(`   🌐 Endereço da Aplicação: http://localhost:${PORT}`);
    console.log(`   📂 Banco Local JSON:      ${USERS_FILE}`);
    console.log(`======================================================`);

    // Inicializa conexão e sincronização com Supabase
    await initSupabase();

    // Observa alterações no arquivo .env para conexão automática instantânea ao salvar
    const envPath = path.join(__dirname, '.env');
    if (fs.existsSync(envPath)) {
        let envTimeout = null;
        fs.watch(envPath, () => {
            clearTimeout(envTimeout);
            envTimeout = setTimeout(async () => {
                console.log('\n[Supabase] 🔄 Arquivo .env alterado! Recarregando credenciais...');
                require('dotenv').config({ override: true });
                await initSupabase();
            }, 600);
        });
    }
});
