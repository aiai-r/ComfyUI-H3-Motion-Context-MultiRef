import { app } from "../../scripts/app.js";

const NODE_NAME = "MiniMaxH3ExtensionStartMode";
const START_T2V_I2V = "start with T2V/I2V";
const SOURCE_GROUP_PREFIX = "START SOURCE VIDEO";
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

function modeWidget(node) {
    return node.widgets?.find((widget) => widget.name === "mode") ?? node.widgets?.[0];
}

function sourceVideoGroup(graph) {
    const groups = graph?._groups ?? graph?.groups ?? [];
    return groups.find((group) => String(group.title ?? "").startsWith(SOURCE_GROUP_PREFIX));
}

function nodeInsideGroup(node, group) {
    if (!node?.pos || !group) return false;
    const bounds = group._bounding ?? group.bounding;
    if (!bounds || bounds.length < 4) return false;
    const [gx, gy, gw, gh] = bounds;
    const [x, y] = node.pos;
    return x >= gx && x <= gx + gw && y >= gy && y <= gy + gh;
}

function applyStartMode(node) {
    const graph = node.graph ?? app.graph;
    const widget = modeWidget(node);
    if (!graph || !widget) return;

    const group = sourceVideoGroup(graph);
    if (!group) return;

    const bypassSourceVideo = String(widget.value) === START_T2V_I2V;
    const desiredMode = bypassSourceVideo ? MODE_BYPASS : MODE_ALWAYS;

    for (const candidate of graph._nodes ?? []) {
        if (!nodeInsideGroup(candidate, group)) continue;
        if (candidate.mode === desiredMode) continue;
        candidate.mode = desiredMode;
        candidate.setDirtyCanvas?.(true, true);
    }

    node.setDirtyCanvas?.(true, true);
}

function bindModeWidget(node) {
    const widget = modeWidget(node);
    if (!widget) return;

    if (!widget._h3StartModeBound) {
        widget._h3StartModeBound = true;
        const originalCallback = widget.callback;
        widget.callback = function(value, ...args) {
            const result = originalCallback?.call(this, value, ...args);
            queueMicrotask(() => applyStartMode(node));
            return result;
        };
    }

    queueMicrotask(() => applyStartMode(node));
}

app.registerExtension({
    name: "seitanism.H3ExtensionStartModeSourceGroup",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(...args) {
            const result = originalCreated?.apply(this, args);
            bindModeWidget(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(...args) {
            const result = originalConfigure?.apply(this, args);
            queueMicrotask(() => bindModeWidget(this));
            return result;
        };

        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(...args) {
            applyStartMode(this);
            return originalSerialize?.apply(this, args);
        };
    },
});
