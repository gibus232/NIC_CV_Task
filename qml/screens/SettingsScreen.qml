import QtQuick
import QtQuick.Controls
import "../components"

Item {
    id: root

    AeroHeader {
        id: hdr
        anchors { top: parent.top; left: parent.left; right: parent.right }
        title: "Настройки"
    }

    Flickable {
        anchors { top: hdr.bottom; left: parent.left; right: parent.right; bottom: parent.bottom }
        contentHeight: settingsCol.height + 40; clip: true

        Column {
            id: settingsCol
            width: parent.width - 32
            anchors { top: parent.top; topMargin: 16; horizontalCenter: parent.horizontalCenter }
            spacing: 12

            // Current user card
            Rectangle {
                width: parent.width; height: 90; radius: 18
                gradient: Gradient { orientation: Gradient.Horizontal; GradientStop { position: 0.0; color: "#0D47A1" }; GradientStop { position: 1.0; color: "#1565C0" } }
                border.color: Qt.rgba(1,1,1,0.28); border.width: 1

                Rectangle { anchors { top: parent.top; left: parent.left; right: parent.right; margins: 1 }; height: parent.height*0.45; radius: parent.radius; gradient: Gradient { GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.18) }; GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) } } }

                Row {
                    anchors { fill: parent; leftMargin: 16; rightMargin: 16 }; spacing: 14
                    Rectangle {
                        width: 58; height: 58; radius: 29; anchors.verticalCenter: parent.verticalCenter
                        gradient: Gradient { GradientStop { position: 0.0; color: Qt.lighter(userManager.currentUserColor,1.4) }; GradientStop { position: 1.0; color: userManager.currentUserColor } }
                        border.color: Qt.rgba(1,1,1,0.45); border.width: 2
                        Text { anchors.centerIn: parent; text: userManager.currentUserName.length>0?userManager.currentUserName[0].toUpperCase():"?"; color: "white"; font.pixelSize: 26; font.weight: Font.Bold }
                    }
                    Column {
                        anchors.verticalCenter: parent.verticalCenter; spacing: 4
                        Text { text: userManager.currentUserName; color: "white"; font.pixelSize: 18; font.weight: Font.Medium }
                        Text { text: "Текущий аккаунт"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }
                    }
                }
            }

            // Section: Account
            Text { text: "Аккаунт"; color: Qt.rgba(1,1,1,0.50); font.pixelSize: 12; leftPadding: 4 }

            SettingsRow {
                width: parent.width; label: "Выйти из аккаунта"; icon: "⏎"
                onRowClicked: userManager.logout()
            }

            // Section: About
            Text { text: "О приложении"; color: Qt.rgba(1,1,1,0.50); font.pixelSize: 12; leftPadding: 4 }

            Rectangle {
                width: parent.width; height: 72; radius: 16
                color: Qt.rgba(1,1,1,0.07); border.color: Qt.rgba(1,1,1,0.14); border.width: 1
                Column {
                    anchors.centerIn: parent; spacing: 4
                    Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Семейный Бюджет"; color: "white"; font.pixelSize: 16; font.weight: Font.Medium }
                    Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Версия 1.0.0  •  Qt 6 • C++"; color: Qt.rgba(1,1,1,0.45); font.pixelSize: 12 }
                }
            }
        }
    }

    // Inline SettingsRow component
    component SettingsRow: Rectangle {
        property string label: ""
        property string icon: ""
        signal rowClicked()

        height: 52; radius: 14
        color: Qt.rgba(1,1,1,0.07); border.color: Qt.rgba(1,1,1,0.14); border.width: 1

        Row {
            anchors { fill: parent; leftMargin: 16; rightMargin: 16 }; spacing: 12
            Text { anchors.verticalCenter: parent.verticalCenter; text: parent.parent.icon; color: "white"; font.pixelSize: 20; width: 28 }
            Text { anchors.verticalCenter: parent.verticalCenter; text: parent.parent.label; color: "white"; font.pixelSize: 15 }
        }
        Text { anchors { right: parent.right; rightMargin: 16; verticalCenter: parent.verticalCenter }; text: "›"; color: Qt.rgba(1,1,1,0.40); font.pixelSize: 22 }

        scale: sma.pressed ? 0.97 : 1.0
        Behavior on scale { NumberAnimation { duration: 80 } }
        MouseArea { id: sma; anchors.fill: parent; onClicked: parent.rowClicked() }
    }
}
