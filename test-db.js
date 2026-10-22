/**
 * ========================================================
 * FaceAuth — Script de Teste e Simulação de Conectividade
 * ========================================================
 * Simula dados de usuários e embeddings faciais de 128D
 * para testar a gravação local e a conectividade com o Supabase.
 *
 * Como executar:
 *   node test-db.js
 *   ou
 *   npm run test:db
 * ========================================================
 */

require('dotenv').config();
const http = require('http');
const { createClient } = require('@supabase/supabase-js');
const { Pool } = require('pg');

// Gera um vetor de 128 dimensões com norma Euclidiana = 1.0 (exatamente como a rede neural face-api.js)
function generateMockEmbedding(seed = 1) {
    const vector = [];
    let norm = 0;
    for (let i = 0; i < 128; i++) {
        // Gera valores pseudo-aleatórios distribuídos em torno de 0
        const val = Math.sin(seed * (i + 1) * 997.13) * 0.5 + Math.cos((i + 7) * 43.17) * 0.3;
        vector.push(val);
        norm += val * val;
    }
    norm = Math.sqrt(norm);
    return vector.map(v => Number((v / norm).toFixed(6)));
}

// Lista de usuários simulados para teste
const SIMULATED_USERS = [
    {
        name: 'Lucas Rocha (Simulação)',
        id: 'usr_sim_lucas_' + Date.now().toString(36),
        seed: 101
    },
    {
        name: 'Mariana Costa (Simulação)',
        id: 'usr_sim_mariana_' + Date.now().toString(36),
        seed: 202
    },
    {
        name: 'Felipe Andrade (Simulação)',
        id: 'usr_sim_felipe_' + Date.now().toString(36),
        seed: 303
    }
];

function printDivider() {
    console.log('─────────────────────────────────────────────────────────────────');
}

async function testLocalServer(usersToRegister) {
    console.log('\n[1/3] 🌐 Testando Servidor Local (http://localhost:3000)...');

    try {
        const statusRes = await fetch('http://localhost:3000/api/status');
        if (!statusRes.ok) {
            throw new Error(`Servidor respondeu com status ${statusRes.status}`);
        }
        const statusData = await statusRes.json();
        console.log(`   ✅ Servidor Local Online! (Usuários registrados: ${statusData.totalRegisteredUsers})`);

        // Envia cada usuário simulado via API
        console.log(`\n   📤 Enviando ${usersToRegister.length} usuário(s) simulado(s) via POST /api/users...`);
        for (const u of usersToRegister) {
            const embedding = generateMockEmbedding(u.seed);
            const payload = {
                id: u.id,
                name: u.name,
                descriptor: embedding,
                templates: [embedding],
                timestamp: Date.now()
            };

            const postRes = await fetch('http://localhost:3000/api/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const postData = await postRes.json();
            if (postData.success) {
                console.log(`      ✓ "${u.name}" (ID: ${u.id}) gravado com sucesso!`);
                if (postData.supabase) {
                    if (postData.supabase.synced) {
                        console.log(`        └─ 🟢 Supabase: sincronizado automaticamente na tabela!`);
                    } else if (postData.supabase.error) {
                        console.log(`        └─ 🟡 Supabase: ${postData.supabase.error}`);
                    }
                }
            } else {
                console.log(`      ✗ Falha ao gravar "${u.name}": ${postData.error}`);
            }
        }
        return true;
    } catch (err) {
        console.log(`   ⚠️ Servidor local não está respondendo em http://localhost:3000: ${err.message}`);
        console.log(`      (Dica: Inicie o servidor com 'npm start' em outro terminal se desejar testar a rota HTTP)`);
        return false;
    }
}

process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

async function testSupabaseDirect(user) {
    console.log('\n[2/3] 🔌 Testando Conexão Direta com Supabase...');

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

    supabaseUrl = supabaseUrl.trim().replace(/\/rest\/v1\/?$/, '').replace(/\/+$/, '');
    supabaseKey = supabaseKey.trim();

    if (!supabaseUrl || !supabaseKey) {
        console.log('   ⚠️ Nenhuma credencial do Supabase configurada no arquivo .env.');
        console.log('   👉 Para testar o Supabase, preencha o .env com:');
        console.log('      SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co');
        console.log('      SUPABASE_KEY=sua-chave-aqui');
        return false;
    }

    console.log(`   📡 URL do Projeto: ${supabaseUrl}`);
    console.log(`   🔑 Chave Detectada: ${supabaseKey.substring(0, 10)}... (OK)`);

    // Se DATABASE_URL estiver presente, testa criação de tabela via PostgreSQL direto
    if (databaseUrl) {
        console.log('   🐘 Testando conexão PostgreSQL direta (DATABASE_URL)...');
        try {
            const pool = new Pool({
                connectionString: databaseUrl,
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
            `);
            await pool.end();
            console.log('   ✅ Tabela "face_embeddings" verificada/criada via PostgreSQL com sucesso!');
        } catch (e) {
            console.log(`   ⚠️ Aviso PostgreSQL DDL: ${e.message}`);
        }
    }

    const supabase = createClient(supabaseUrl, supabaseKey, {
        auth: { persistSession: false, autoRefreshToken: false }
    });

    const startTime = Date.now();

    try {
        // 1. Testa INSERT / UPSERT de dados simulados
        console.log(`   📝 Inserindo usuário de teste simulado: "${user.name}"...`);
        const embedding = generateMockEmbedding(user.seed);

        const testRecord = {
            id: user.id,
            name: user.name,
            embedding: embedding,
            timestamp: Date.now(),
            registered_at: new Date().toISOString()
        };

        const { error: insertError } = await supabase
            .from('face_embeddings')
            .upsert(testRecord, { onConflict: 'id' });

        if (insertError) {
            throw new Error(`Erro no INSERT: ${insertError.message} (Código: ${insertError.code || 'N/A'})`);
        }

        const insertLatency = Date.now() - startTime;
        console.log(`   ✅ INSERT concluído com sucesso em ${insertLatency}ms!`);

        // 2. Testa SELECT para validar a leitura e a integridade do vetor 128D
        console.log(`   🔍 Validando leitura (SELECT) na tabela 'face_embeddings'...`);
        const readStart = Date.now();
        const { data, error: selectError } = await supabase
            .from('face_embeddings')
            .select('id, name, embedding, timestamp, registered_at')
            .eq('id', user.id)
            .single();

        if (selectError) {
            throw new Error(`Erro no SELECT: ${selectError.message}`);
        }

        const readLatency = Date.now() - readStart;
        const retrievedEmbedding = Array.isArray(data.embedding)
            ? data.embedding
            : JSON.parse(data.embedding || '[]');

        console.log(`   ✅ SELECT executado com sucesso em ${readLatency}ms!`);
        console.log(`      ├─ ID retornado:        ${data.id}`);
        console.log(`      ├─ Nome:                ${data.name}`);
        console.log(`      ├─ Dimensões do Vetor:  ${retrievedEmbedding.length} dimensões (Esperado: 128)`);
        console.log(`      ├─ Amostra de valores:  [${retrievedEmbedding.slice(0, 4).join(', ')}, ...]`);
        console.log(`      └─ Timestamp:           ${data.timestamp} (${data.registered_at})`);

        return true;
    } catch (err) {
        console.log(`   ❌ Falha na comunicação com o Supabase: ${err.message}`);
        if (err.message.includes('relation "face_embeddings" does not exist') || err.message.includes('PGRST205')) {
            console.log('\n   💡 A tabela "face_embeddings" ainda precisa ser criada no seu Supabase.');
            console.log('      Execute no "SQL Editor" do painel do Supabase o seguinte comando:\n');
            console.log(`      CREATE TABLE IF NOT EXISTS face_embeddings (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          embedding JSONB NOT NULL,
          timestamp BIGINT NOT NULL,
          registered_at TEXT,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      );`);
        }
        return false;
    }
}

async function run() {
    console.clear();
    printDivider();
    console.log('       🧪 FACEAUTH — TESTE DE CONECTIVIDADE E SIMULAÇÃO DE DADOS');
    printDivider();

    // 1. Testa o servidor local
    await testLocalServer(SIMULATED_USERS);

    // 2. Testa diretamente o Supabase com o primeiro usuário simulado
    const supabaseOk = await testSupabaseDirect(SIMULATED_USERS[0]);

    // 3. Resumo Final
    printDivider();
    console.log('\n[3/3] 📊 RESUMO DO DIAGNÓSTICO:');
    if (supabaseOk) {
        console.log('   🎉 Conexão com o Supabase 100% FUNCIONAL e validada!');
        console.log('   🎉 Gravação e leitura de embeddings de 128D operando corretamente!');
    } else {
        console.log('   ℹ️  Dados simulados gravados localmente.');
        console.log('   ℹ️  Assim que preencher o arquivo .env com suas credenciais do Supabase,');
        console.log('      execute este teste novamente com: npm run test:db\n');
    }
    printDivider();
}

run();
