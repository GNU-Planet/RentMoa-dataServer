# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class DetachedHouseTransaction(models.Model):
    index = models.BigIntegerField(blank=True, null=True)
    지역코드 = models.TextField(blank=True, null=True)
    법정동 = models.TextField(blank=True, null=True)
    지번 = models.TextField(blank=True, null=True)
    주택유형 = models.TextField(blank=True, null=True)
    건축년도 = models.BigIntegerField(blank=True, null=True)
    대지면적 = models.FloatField(blank=True, null=True)
    연면적 = models.FloatField(blank=True, null=True)
    년 = models.BigIntegerField(blank=True, null=True)
    월 = models.BigIntegerField(blank=True, null=True)
    일 = models.BigIntegerField(blank=True, null=True)
    거래금액 = models.BigIntegerField(blank=True, null=True)
    거래유형 = models.TextField(blank=True, null=True)
    중개사소재지 = models.TextField(blank=True, null=True)
    해제사유발생일 = models.TextField(blank=True, null=True)
    해제여부 = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = '단독다가구매매'


class DetachedHouseRent(models.Model):
    index = models.BigIntegerField(blank=True, null=True)
    지역코드 = models.TextField(blank=True, null=True)
    법정동 = models.TextField(blank=True, null=True)
    건축년도 = models.BigIntegerField(blank=True, null=True)
    계약면적 = models.FloatField(blank=True, null=True)
    년 = models.BigIntegerField(blank=True, null=True)
    월 = models.BigIntegerField(blank=True, null=True)
    일 = models.BigIntegerField(blank=True, null=True)
    보증금액 = models.BigIntegerField(blank=True, null=True)
    월세금액 = models.BigIntegerField(blank=True, null=True)
    계약구분 = models.TextField(blank=True, null=True)
    계약기간 = models.TextField(blank=True, null=True)
    갱신요구권사용 = models.TextField(blank=True, null=True)
    종전계약보증금 = models.BigIntegerField(blank=True, null=True)
    종전계약월세 = models.BigIntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = '단독다가구전월세'
