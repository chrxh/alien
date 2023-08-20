<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $success = checkPw($db, $userName, $pw);

    if ($success) {
        $success = $db->query("UPDATE user SET FLAGS=1, TIMESTAMP=CURRENT_TIMESTAMP, TIME_SPENT = COALESCE(TIME_SPENT + 1, 1) WHERE NAME='".addslashes($userName)."'");
    }

    $db->commit();
    $db->close();
?>
