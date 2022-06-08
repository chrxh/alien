<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $activationCode = $_POST["activationCode"];

    if (!checkPwAndActivationCode($db, $userName, $pw, $activationCode)) {
        echo json_encode(["result"=>false]);
        $db->close();
        exit;
    }

    $success = false;
    if ($db->query("UPDATE user SET ACTIVATION_CODE='' where NAME='" . addslashes($userName) . "'")) {
        $success = true;

    }
    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();
?>