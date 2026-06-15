import QtQuick
import QtQuick.Controls
import FamilyBudget

Item {
    id: root

    AeroHeader {
        id: hdr
        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
        title: "Семья"
        showAction: true; actionIcon: "+"
        onActionClicked: addPopup.open()
    }

    ListView {
        id: userList
        anchors.top: hdr.bottom; anchors.topMargin: 12; anchors.bottom: parent.bottom; anchors.bottomMargin: 8; anchors.left: parent.left; anchors.leftMargin: 16; anchors.right: parent.right; anchors.rightMargin: 16
        spacing: 12; clip: true
        model: userManager.getUsers()

        delegate: Rectangle {
            width: userList.width; height: 90; radius: 18
            color: modelData.id === userManager.currentUserId ? Qt.rgba(0.2,0.5,1.0,0.18) : Qt.rgba(1,1,1,0.08)
            border.color: modelData.id === userManager.currentUserId ? Qt.rgba(0.4,0.7,1.0,0.40) : Qt.rgba(1,1,1,0.18)
            border.width: 1

            Rectangle {
                anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1
                height: parent.height * 0.45; radius: parent.radius
                gradient: Gradient { GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.08) }
                        GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) } }
            }

            Row {
                anchors.fill: parent; anchors.leftMargin: 16; anchors.rightMargin: 16; spacing: 14

                Rectangle {
                    width: 56; height: 56; radius: 28; anchors.verticalCenter: parent.verticalCenter
                    gradient: Gradient { GradientStop { position: 0.0; color: Qt.lighter(modelData.color||"#1565C0",1.4) }
                        GradientStop { position: 1.0; color: modelData.color||"#1565C0" } }
                    border.color: Qt.rgba(1,1,1,0.45); border.width: 2
                    Text { anchors.centerIn: parent; text: (modelData.name||"?")[0].toUpperCase(); color: "white"; font.pixelSize: 24; font.weight: Font.Bold }
                }

                Column {
                    anchors.verticalCenter: parent.verticalCenter; spacing: 4; width: parent.width - 56 - 60 - 28
                    Text { text: modelData.name; color: "white"; font.pixelSize: 17; font.weight: Font.Medium }
                    Text {
                        text: modelData.id === userManager.currentUserId ? "★ Текущий аккаунт" :
                              (userManager.hasPin(modelData.id) ? "🔒 PIN защищён" : "Без PIN")
                        color: modelData.id === userManager.currentUserId ? "#4FC3F7" : Qt.rgba(1,1,1,0.55)
                        font.pixelSize: 13
                    }
                }

                // Delete button (can't delete current user)
                Rectangle {
                    visible: modelData.id !== userManager.currentUserId
                    anchors.verticalCenter: parent.verticalCenter
                    width: 40; height: 40; radius: 20
                    color: Qt.rgba(0.8,0.2,0.2,0.20); border.color: Qt.rgba(1,0.4,0.4,0.35); border.width: 1
                    Text { anchors.centerIn: parent; text: "✕"; color: "#EF5350"; font.pixelSize: 18 }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            userManager.deleteUser(modelData.id)
                            userList.model = userManager.getUsers()
                        }
                    }
                }
            }
        }
    }

    Connections { target: userManager; function onUsersChanged() { userList.model = userManager.getUsers() } }

    // Add member popup (same as LoginScreen's addPopup)
    Popup {
        id: addPopup
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 340)
        modal: true; dim: true
        onOpened: { nameIn.text = ""; pinIn.text = ""; chosenColor = "#1565C0" }
        property string chosenColor: "#1565C0"

        background: Rectangle { radius: 24; color: Qt.rgba(0.05,0.10,0.25,0.96); border.color: Qt.rgba(1,1,1,0.30); border.width: 1; Rectangle { anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; anchors.margins: 1; height: 48; radius: parent.radius; gradient: Gradient { GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.12) }
                        GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) } } } }
        Overlay.modal: Rectangle { color: Qt.rgba(0,0,0,0.60) }

        Column {
            width: parent.width; spacing: 14; padding: 24
            Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Новый член семьи"; color: "white"; font.pixelSize: 18; font.weight: Font.Medium }
            Rectangle { width: parent.width-48; height: 46; radius: 12; anchors.horizontalCenter: parent.horizontalCenter; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.25); border.width: 1; TextInput { id: nameIn; anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14; verticalAlignment: TextInput.AlignVCenter; color: "white"; font.pixelSize: 15; placeholderText: "Имя"} }
            Rectangle { width: parent.width-48; height: 46; radius: 12; anchors.horizontalCenter: parent.horizontalCenter; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.25); border.width: 1; TextInput { id: pinIn; anchors.fill: parent; anchors.leftMargin: 14; anchors.rightMargin: 14; verticalAlignment: TextInput.AlignVCenter; color: "white"; font.pixelSize: 15; echoMode: TextInput.Password; inputMethodHints: Qt.ImhDigitsOnly; maximumLength: 4; placeholderText: "PIN (необязательно)"} }
            Text { text: "Цвет:"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }
            Flow { width: parent.width-48; anchors.horizontalCenter: parent.horizontalCenter; spacing: 10
                Repeater { model: ["#F44336","#E91E63","#9C27B0","#3F51B5","#2196F3","#00BCD4","#4CAF50","#FF9800","#795548","#607D8B"]
                    delegate: Rectangle { width: 36; height: 36; radius: 18; color: modelData; border.color: addPopup.chosenColor===modelData?"white":"transparent"; border.width: 3; MouseArea { anchors.fill: parent; onClicked: addPopup.chosenColor=modelData } } }
            }
            Row { anchors.horizontalCenter: parent.horizontalCenter; spacing: 12
                Rectangle { width: 100; height: 44; radius: 22; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1; Text { anchors.centerIn: parent; text: "Отмена"; color: "white"; font.pixelSize: 14 }
                        MouseArea { anchors.fill: parent; onClicked: addPopup.close() } }
                Rectangle { width: 130; height: 44; radius: 22; gradient: Gradient { orientation: Gradient.Horizontal; GradientStop { position: 0.0; color: "#1E88E5" }
                        GradientStop { position: 1.0; color: "#1565C0" } }
                    border.color: Qt.rgba(1,1,1,0.35); border.width: 1
                    Text { anchors.centerIn: parent; text: "Добавить"; color: "white"; font.pixelSize: 14; font.weight: Font.Medium }
                    MouseArea { anchors.fill: parent; onClicked: { var n=nameIn.text.trim(); if(n!==""){ userManager.createUser(n,pinIn.text,addPopup.chosenColor,"person"); addPopup.close() } } }
                }
            }
        }
    }
}
