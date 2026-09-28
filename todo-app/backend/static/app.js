(function () {
  "use strict";

  var API = "/api/todos";

  var listEl = document.getElementById("todo-list");
  var formEl = document.getElementById("add-form");
  var inputEl = document.getElementById("add-input");
  var addBtn = document.getElementById("add-btn");
  var emptyEl = document.getElementById("empty-state");
  var countEl = document.getElementById("count-text");
  var toastEl = document.getElementById("toast");

  var todos = [];
  var toastTimer = null;

  // ---------- 工具 ----------
  function showToast(msg) {
    toastEl.textContent = msg;
    toastEl.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.hidden = true;
    }, 2200);
  }

  function fail(msg) {
    console.error(msg);
    showToast(msg);
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ---------- 渲染 ----------
  function render() {
    listEl.innerHTML = "";

    var done = todos.filter(function (t) {
      return t.completed;
    }).length;
    countEl.textContent = "共 " + todos.length + " 项，已完成 " + done + " 项";

    emptyEl.hidden = todos.length !== 0;

    todos.forEach(function (todo) {
      var li = document.createElement("li");
      li.className = "todo-item" + (todo.completed ? " completed" : "");
      li.dataset.id = todo.id;

      // checkbox
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !!todo.completed;
      cb.setAttribute("aria-label", "标记完成");
      cb.addEventListener("change", function () {
        toggleComplete(todo, cb.checked, cb);
      });

      // text / edit input
      var span = document.createElement("span");
      span.className = "todo-text";
      span.textContent = todo.text;
      span.title = "双击编辑";
      span.addEventListener("dblclick", function () {
        startEdit(li, todo, span);
      });

      // delete
      var del = document.createElement("button");
      del.className = "delete-btn";
      del.textContent = "×";
      del.setAttribute("aria-label", "删除");
      del.addEventListener("click", function () {
        removeTodo(todo, del);
      });

      li.appendChild(cb);
      li.appendChild(span);
      li.appendChild(del);
      listEl.appendChild(li);
    });
  }

  // ---------- API ----------
  function loadTodos() {
    countEl.textContent = "加载中…";
    fetch(API)
      .then(function (res) {
        if (!res.ok) throw new Error("GET " + res.status);
        return res.json();
      })
      .then(function (data) {
        todos = Array.isArray(data) ? data : [];
        render();
      })
      .catch(function (err) {
        fail("加载失败：" + err.message);
        countEl.textContent = "加载失败";
        emptyEl.hidden = true;
      });
  }

  function addTodo(text) {
    addBtn.disabled = true;
    inputEl.disabled = true;
    fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then(function (res) {
        if (res.status !== 201 && !res.ok) throw new Error("POST " + res.status);
        return res.json();
      })
      .then(function (created) {
        todos.push(created);
        render();
        inputEl.value = "";
        inputEl.focus();
      })
      .catch(function (err) {
        fail("添加失败：" + err.message);
      })
      .finally(function () {
        addBtn.disabled = false;
        inputEl.disabled = false;
      });
  }

  function toggleComplete(todo, checked, checkboxEl) {
    checkboxEl.disabled = true;
    fetch(API + "/" + todo.id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ completed: checked }),
    })
      .then(function (res) {
        if (!res.ok) throw new Error("PUT " + res.status);
        return res.json();
      })
      .then(function (updated) {
        todo.completed = updated.completed;
        render();
      })
      .catch(function (err) {
        fail("更新失败：" + err.message);
        checkboxEl.checked = !checked;
        checkboxEl.disabled = false;
      });
  }

  function saveEdit(todo, newText, onDone) {
    fetch(API + "/" + todo.id, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: newText }),
    })
      .then(function (res) {
        if (!res.ok) throw new Error("PUT " + res.status);
        return res.json();
      })
      .then(function (updated) {
        todo.text = updated.text;
        render();
      })
      .catch(function (err) {
        fail("保存失败：" + err.message);
      })
      .finally(onDone);
  }

  function removeTodo(todo, btnEl) {
    btnEl.disabled = true;
    fetch(API + "/" + todo.id, { method: "DELETE" })
      .then(function (res) {
        if (res.status !== 204 && !res.ok) throw new Error("DELETE " + res.status);
        todos = todos.filter(function (t) {
          return t.id !== todo.id;
        });
        render();
      })
      .catch(function (err) {
        fail("删除失败：" + err.message);
        btnEl.disabled = false;
      });
  }

  // ---------- 编辑态 ----------
  function startEdit(li, todo, span) {
    if (li.querySelector(".todo-edit-input")) return;

    var input = document.createElement("input");
    input.type = "text";
    input.className = "todo-edit-input";
    input.value = todo.text;
    input.maxLength = 200;

    li.replaceChild(input, span);
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);

    var done = false;

    function commit() {
      if (done) return;
      done = true;
      var v = input.value.trim();
      if (!v) {
        // 空内容：取消编辑，恢复原文
        render();
        return;
      }
      if (v === todo.text) {
        render();
        return;
      }
      input.disabled = true;
      saveEdit(todo, v, function () {});
    }

    function cancel() {
      if (done) return;
      done = true;
      render();
    }

    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        commit();
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancel();
      }
    });
    input.addEventListener("blur", commit);
  }

  // ---------- 事件绑定 ----------
  formEl.addEventListener("submit", function (e) {
    e.preventDefault();
    var v = inputEl.value.trim();
    if (!v) {
      showToast("请输入内容");
      return;
    }
    addTodo(v);
  });

  // ---------- 启动 ----------
  loadTodos();
})();
