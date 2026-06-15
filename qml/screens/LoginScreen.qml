import QtQuick
import QtQuick.Controls
import FamilyBudget

Item {
    id: root
    property int  pendingUserId:   -1
    property string pendingName:   ""

    // ---- Logo & title ----
    Column {
        anchors.top: parent.top; anchors.topMargin: 64; anchors.horizontalCenter: parent.horizontalCenter
        spacing: 10

        Rectangle {
            anchors.horizontalCenter: parent.horizontalCenter
            width: 82; height: 82; radius: 41
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#42A5F5" }
                GradientStop { position: 1.0; color: "#1565C0" }
            }
            border.color: Qt.rgba(1,1,1,0.45); border.width: 2

            Rectangle {
                anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                height: parent.height * 0.48; radius: parent.radius
                gradient: Gradient {
                    GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.30) }
                    GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
                }
            }
            Text { anchors.centerIn: parent; text: "₽"; color: "white"; font.pixelSize: 38; font.weight: Font.Bold }
        }

        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Семейный Бюджет"; color: "white"; font.pixelSize: 26; font.weight: Font.Bold }
        Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Выберите аккаунт"; color: Qt.rgba(1,1,1,0.60); font.pixelSize: 15 }
    }

    // ---- User list ----
    ListView {
        id: userList
        anchors.top: parent.top; anchors.topMargin: 230; anchors.bottom: addBtn.top; anchors.bottomMargin: 16; anchors.left: parent.left; anchors.leftMargin: 20; anchors.right: parent.right; anchors.rightMargin: 20
        spacing: 12; clip: true
        model: userManager.getUsers()

        delegate: Rectangle {
            width: userList.width; height: 80; radius: 18
            color: Qt.rgba(1,1,1,0.09)
            border.color: Qt.rgba(1,1,1,0.22); border.width: 1

            Rectangle {
                anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                height: parent.height * 0.45; radius: parent.radius
                gradient: Gradient {
                    GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.10) }
                    GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
                }
            }

            Row {
                anchors.fill: parent; anchors.leftMargin: 16; anchors.rightMargin: 16; spacing: 14

                Rectangle {
                    width: 52; height: 52; radius: 26
                    anchors.verticalCenter: parent.verticalCenter
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: Qt.lighter(modelData.color || "#1565C0", 1.4) }
                        GradientStop { position: 1.0; color: modelData.color || "#1565C0" }
                    }
                    border.color: Qt.rgba(1,1,1,0.45); border.width: 2
                    Text { anchors.centerIn: parent; text: (modelData.name||"?")[0].toUpperCase(); color: "white"; font.pixelSize: 22; font.weight: Font.Bold }
                }

                Column {
                    anchors.verticalCenter: parent.verticalCenter; spacing: 4
                    Text { text: modelData.name || ""; color: "white"; font.pixelSize: 18; font.weight: Font.Medium }
                    Text {
                        text: userManager.hasPin(modelData.id) ? "🔒 Защищён PIN" : "Без пароля"
                        color: Qt.rgba(1,1,1,0.55); font.pixelSize: 13
                    }
                }
            }

            scale: cardMa.pressed ? 0.97 : 1.0
            Behavior on scale { NumberAnimation { duration: 80 } }

            MouseArea {
                id: cardMa; anchors.fill: parent
                onClicked: {
                    root.pendingUserId = modelData.id
                    root.pendingName   = modelData.name
                    if (userManager.hasPin(modelData.id)) pinPopup.open()
                    else userManager.login(modelData.id, "")
                }
            }
        }
    }

    // ---- Add account button ----
    Rectangle {
        id: addBtn
        anchors.bottom: parent.bottom; anchors.bottomMargin: 44; anchors.horizontalCenter: parent.horizontalCenter
        width: 240; height: 52; radius: 26
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "#1E88E5" }
            GradientStop { position: 1.0; color: "#0D47A1" }
        }
        border.color: Qt.rgba(1,1,1,0.35); border.width: 1

        Rectangle {
            anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
            height: parent.height * 0.50; radius: parent.radius
            gradient: Gradient {
                GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.25) }
                GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
            }
        }
        Text { anchors.centerIn: parent; text: "+ Добавить аккаунт"; color: "white"; font.pixelSize: 16; font.weight: Font.Medium }

        scale: addMa.pressed ? 0.96 : 1.0
        Behavior on scale { NumberAnimation { duration: 80 } }
        MouseArea { id: addMa; anchors.fill: parent; onClicked: addPopup.open() }
    }

    Connections { target: userManager; function onUsersChanged() { userList.model = userManager.getUsers() } }

    // ---- PIN popup ----
    Popup {
        id: pinPopup
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 320)
        modal: true; dim: true
        onOpened: pinPad.pin = ""

        background: Rectangle {
            radius: 24; color: Qt.rgba(0.05,0.10,0.25,0.96)
            border.color: Qt.rgba(1,1,1,0.30); border.width: 1
            Rectangle {
                anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                height: 48; radius: parent.radius
                gradient: Gradient {
                    GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.12) }
                    GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
                }
            }
        }
        Overlay.modal: Rectangle { color: Qt.rgba(0,0,0,0.60) }

        Column {
            width: parent.width; spacing: 16; padding: 24

            Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Введите PIN"; color: "white"; font.pixelSize: 18; font.weight: Font.Medium }
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: root.pendingName; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 14 }

            PinPad {
                id: pinPad; anchors.horizontalCenter: parent.horizontalCenter
                onPinComplete: function(p) {
                    if (userManager.login(root.pendingUserId, p)) pinPopup.close()
                    else { pin = ""; errAnim.start() }
                }
            }

            Text {
                id: errTxt; anchors.horizontalCenter: parent.horizontalCenter
                text: "Неверный PIN"; color: "#EF5350"; font.pixelSize: 14; opacity: 0
                NumberAnimation on opacity { id: errAnim; from: 1; to: 0; duration: 1600 }
            }

            Rectangle {
                anchors.horizontalCenter: parent.horizontalCenter
                width: 110; height: 40; radius: 20
                color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
                Text { anchors.centerIn: parent; text: "Отмена"; color: "white"; font.pixelSize: 14 }
                MouseArea { anchors.fill: parent; onClicked: { pinPad.pin = ""; pinPopup.close() } }
            }
        }
    }

    // ---- Add user popup ----
    Popup {
        id: addPopup
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 340)
        modal: true; dim: true
        onOpened: { nameIn.text = ""; pinIn.text = ""; chosenColor = "#1565C0" }

        property string chosenColor: "#1565C0"

        background: Rectangle {
            radius: 24; color: Qt.rgba(0.05,0.10,0.25,0.96)
            border.color: Qt.rgba(1,1,1,0.30); border.width: 1
            Rectangle {
                anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                height: 48; radius: parent.radius
                gradient: Gradient {
                    GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.12) }
                    GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
                }
            }
        }
        Overlay.modal: Rectangle { color: Qt.rgba(0,0,0,0.60) }
        enter: Transition { NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 250 } }
        exit:  Transition { NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 200 } }

        Column {
            width: parent.width; spacing: 14; padding: 24

            Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Новый аккаунт"; color: "white"; font.pixelSize: 18; font.weight: Font.Medium }

            Rectangle {
                width: parent.width - 48; height: 46; radius: 12; anchors.horizontalCenter: parent.horizontalCenter
                color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.25); border.width: 1
                TextInput { id: nameIn; anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14; verticalAlignment: TextInput.AlignVCenter; color: "white"; font.pixelSize: 15; placeholderText: "Имя"; placeholderTextColor: Qt.rgba(1,1,1,0.38) }
            }

            Rectangle {
                width: parent.width - 48; height: 46; radius: 12; anchors.horizontalCenter: parent.horizontalCenter
                color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.25); border.width: 1
                TextInput { id: pinIn; anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14; verticalAlignment: TextInput.AlignVCenter; color: "white"; font.pixelSize: 15; echoMode: TextInput.Password; inputMethodHints: Qt.ImhDigitsOnly; maximumLength: 4; placeholderText: "PIN (необязательно)"; placeholderTextColor: Qt.rgba(1,1,1,0.38) }
            }

            Text { text: "Цвет аватара:"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }

            Flow {
                width: parent.width - 48; anchors.horizontalCenter: parent.horizontalCenter; spacing: 10
                Repeater {
                    model: ["#F44336","#E91E63","#9C27B0","#3F51B5","#2196F3","#00BCD4","#4CAF50","#FF9800","#795548","#607D8B"]
                    delegate: Rectangle {
                        width: 36; height: 36; radius: 18; color: modelData
                        border.color: addPopup.chosenColor === modelData ? "white" : "transparent"; border.width: 3
                        MouseArea { anchors.fill: parent; onClicked: addPopup.chosenColor = modelData }
                    }
                }
            }

            Row {
                anchors.horizontalCenter: parent.horizontalCenter; spacing: 12

                Rectangle {
                    width: 100; height: 44; radius: 22
                    color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
                    Text { anchors.centerIn: parent; text: "Отмена"; color: "white"; font.pixelSize: 14 }
                    MouseArea { anchors.fill: parent; onClicked: addPopup.close() }
                }

                Rectangle {
                    width: 130; height: 44; radius: 22
                    gradient: Gradient { orientation: Gradient.Horizontal; GradientStop { position: 0.0; color: "#1E88E5" }
                        GradientStop { position: 1.0; color: "#1565C0" } }
                    border.color: Qt.rgba(1,1,1,0.35); border.width: 1
                    Text { anchors.centerIn: parent; text: "Создать"; color: "white"; font.pixelSize: 14; font.weight: Font.Medium }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            var n = nameIn.text.trim()
                            if (n !== "") {
                                userManager.createUser(n, pinIn.text, addPopup.chosenColor, "person")
                                addPopup.close()
                            }
                        }
                    }
                }
            }
        }
    }
}
