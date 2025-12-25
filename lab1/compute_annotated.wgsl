// Lab 1 — Compute shader (WGSL) — version commentée
// Objectif: data1[i] = data0[i]^2
//
// LIEN PROJET CLOTH:
// - data0/data1 seraient remplacés par positions/velocities/springs/params
// - 1 thread GPU = 1 vertex (souvent)
// - gid.x sert d'index "i" dans tes buffers

// group 0, binding 0: buffer d'entrée (lecture seule)
@group(0) @binding(0)
// déclaration d'un storage buffer en lecture seule contenant un tableau d'entiers 32 bits
// data0 est le nom du buffer dans le shader
// array<i32> indique que c'est un tableau d'entiers 32 bits
var<storage, read> data0: array<i32>;

// group 0, binding 1: buffer de sortie (lecture/écriture)
// déclaration d'un storage buffer en lecture/écriture contenant un tableau d'entiers 32 bits
// data1 est le nom du buffer dans le shader
// array<i32> indique que c'est un tableau d'entiers 32 bits
@group(0) @binding(1)
var<storage, read_write> data1: array<i32>;

// Fonction principale du compute shader
// Chaque thread exécute cette fonction
// workgroup_size(64): chaque groupe de travail contient 64 threads
// c'est quoi thread? un thread est une unité d'exécution sur le GPU, chaque thread exécute le shader de manière indépendante
// compute: indique que c'est une fonction de compute shader
// on peut avoir workgroup_size sur x, y, z (ici on utilise seulement x)?
// oui, on peut définir des tailles de groupe de travail sur les axes x, y, z selon les besoins du calcul 
// du shader. Ici, on utilise seulement l'axe x car le calcul est unidimensionnel.
// chaque thread a un identifiant global (gid) qui permet de savoir quel élément du tableau il doit traiter
@compute @workgroup_size(64)
// point d'entrée du shader
// c'est quoi global_invocation_id?
// global_invocation_id est une variable intégrée qui donne l'identifiant global du thread en cours d'exécution
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Index global du thread (sur l'axe X)
    // gid.x est l'index du thread courant 
    // (valeur entre 0 et N-1 si on dispatch N threads)
    let i: u32 = gid.x;
    // i sert d'index pour accéder aux éléments des buffers data0 et data1
    // on utilise i pour lire data0[i] et écrire data1[i]
    // i est de type u32 (entier non signé 32 bits)
    // on pourrait aussi utiliser gid.y ou gid.z si on avait des workgroups sur y ou z


    // Sécurité: si on dispatch un peu trop, on évite d'écrire hors buffer
    // (Pour le lab on a choisi N multiple de 64 donc ça ne déclenche pas.)
    if (i >= arrayLength(&data0)) {
        return;
    }

    // Calcul jouet
    data1[i] = data0[i] * data0[i];
}
