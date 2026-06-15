import QtQuick
import QtQuick.Controls

ApplicationWindow {
    id: root
    width: 393
    height: 852
    visible: true
    title: "Семейный Бюджет"

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#071221" }
            GradientStop { position: 0.4; color: "#0D3461" }
            GradientStop { position: 1.0; color: "#061831" }
        }
    }

    // Frutiger Aero bokeh circles
    Repeater {
        model: [
            {cx:60,  cy:120, r:90},
            {cx:330, cy:80,  r:110},
            {cx:200, cy:300, r:70},
            {cx:350, cy:430, r:130},
            {cx:40,  cy:560, r:95},
            {cx:270, cy:710, r:80},
            {cx:140, cy:810, r:115}
        ]
        delegate: Rectangle {
            x: modelData.cx - modelData.r
            y: modelData.cy - modelData.r
            width:  modelData.r * 2
            height: modelData.r * 2
            radius: modelData.r
            color: "transparent"
            border.color: Qt.rgba(0.28, 0.62, 1.0, 0.07)
            border.width: 2
        }
    }

    StackView {
        id: stackView
        anchors.fill: parent
        initialItem: userManager.isLoggedIn
            ? Qt.resolvedUrl("screens/DashboardScreen.qml")
            : Qt.resolvedUrl("screens/LoginScreen.qml")

        pushEnter:    Transition { NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 320; easing.type: Easing.OutCubic } }
        pushExit:     Transition { NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 200 } }
        popEnter:     Transition { NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 300 } }
        popExit:      Transition { NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 200 } }
        replaceEnter: Transition { NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 400; easing.type: Easing.OutCubic } }
        replaceExit:  Transition { NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 250 } }
    }

    Connections {
        target: userManager
        function onCurrentUserChanged() {
            if (userManager.isLoggedIn)
                stackView.replace(Qt.resolvedUrl("screens/DashboardScreen.qml"))
            else
                stackView.replace(Qt.resolvedUrl("screens/LoginScreen.qml"))
        }
    }
}
